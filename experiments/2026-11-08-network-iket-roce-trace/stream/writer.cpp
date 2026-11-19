// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <zlib.h>

struct alignas(64) Header {
  uint64_t capacity, tail, head, status, waits, stop, reserved[2];
};
struct Record {
  uint64_t timestamp, grid, kernel, cta;
  uint32_t warp, sm, event, payload, kind, reserved;
};
struct alignas(64) Slot {
  uint64_t sequence;
  Record record;
};
struct DiskRecord {
  uint64_t sequence;
  Record record;
};
static_assert(sizeof(Header) == 64 && sizeof(Slot) == 64 &&
              sizeof(DiskRecord) == 64);
static uint64_t load(uint64_t *p) {
  return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
static void store(uint64_t *p, uint64_t n) {
  __atomic_store_n(p, n, __ATOMIC_RELEASE);
}

struct Writer {
  Header *header;
  Slot *slots;
  std::string path;
  uint64_t chunk_records, delay_us, fail_after, head = 0, chunk = 0, count = 0;
  uint64_t hash = 0;
  std::atomic<uint64_t> written{0}, chunks{0};
  int fd = -1;
  std::thread worker;
  std::string partial, final;
  Writer(void *memory, const char *out, uint64_t bytes, uint64_t delay,
         uint64_t fail)
      : header(static_cast<Header *>(memory)),
        slots(reinterpret_cast<Slot *>(header + 1)), path(out),
        chunk_records((bytes - 64) / 64), delay_us(delay), fail_after(fail) {
    if (bytes < 128 || !header->capacity)
      throw std::runtime_error("invalid stream geometry");
    open_chunk();
    worker = std::thread([this] { run(); });
  }
  void write_all(const void *data, size_t size) {
    auto p = static_cast<const char *>(data);
    while (size) {
      auto n = ::write(fd, p, size);
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0) {
        int code = errno ? errno : EIO;
        store(&header->status, code);
        throw std::runtime_error("trace write failed");
      }
      p += n;
      size -= n;
    }
  }
  void open_chunk() {
    char name[64];
    std::snprintf(name, sizeof(name), "/events-%08llu.bin",
                  static_cast<unsigned long long>(chunk));
    final = path + name;
    partial = final + ".partial";
    fd = ::open(partial.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) {
      store(&header->status, errno);
      throw std::runtime_error("cannot create trace chunk");
    }
    struct FileHeader {
      char magic[8];
      uint32_t version, record_size;
      uint64_t chunk, first;
      uint64_t reserved[4];
    } meta{};
    static_assert(sizeof(meta) == 64);
    std::memcpy(meta.magic, "SPKIKET1", 8);
    meta.version = 2;
    meta.record_size = 64;
    meta.chunk = chunk;
    meta.first = written.load();
    write_all(&meta, sizeof(meta));
    count = 0;
    hash = 0;
  }
  void close_chunk() {
    if (fd < 0)
      return;
    int error = ::fsync(fd) ? errno : 0;
    if (::close(fd) && !error)
      error = errno;
    if (error) {
      fd = -1;
      store(&header->status, error);
      throw std::runtime_error("trace sync failed");
    }
    fd = -1;
    if (::rename(partial.c_str(), final.c_str())) {
      store(&header->status, errno);
      throw std::runtime_error("trace rename failed");
    }
    std::string receipt = final + ".json";
    FILE *f = std::fopen(receipt.c_str(), "wx");
    if (!f) {
      store(&header->status, errno);
      throw std::runtime_error("cannot create chunk receipt");
    }
    std::fprintf(f, "{\"chunk\":%llu,\"records\":%llu,\"crc32\":\"%08llx\"}\n",
                 static_cast<unsigned long long>(chunk),
                 static_cast<unsigned long long>(count),
                 static_cast<unsigned long long>(hash));
    if (std::fclose(f)) {
      store(&header->status, errno ? errno : EIO);
      throw std::runtime_error("chunk receipt failed");
    }
    chunks.fetch_add(1);
    ++chunk;
  }
  void append(const DiskRecord *data, size_t n) {
    if (fail_after && written.load() + n > fail_after) {
      store(&header->status, ENOSPC);
      throw std::runtime_error("injected disk-full error");
    }
    if (delay_us)
      std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
    write_all(data, n * sizeof(DiskRecord));
    hash =
        crc32(static_cast<uLong>(hash), reinterpret_cast<const Bytef *>(data),
              static_cast<uInt>(n * sizeof(DiskRecord)));
    count += n;
    written.fetch_add(n);
  }
  void run() noexcept {
    try {
      std::vector<DiskRecord> batch(1024);
      while (!load(&header->status)) {
        size_t n = 0;
        while (n < batch.size() && n < chunk_records - count) {
          Slot &slot = slots[head % header->capacity];
          if (load(&slot.sequence) != head + 1)
            break;
          batch[n++] = DiskRecord{head, slot.record};
          store(&slot.sequence, head + header->capacity);
          store(&header->head, ++head);
        }
        if (n)
          append(batch.data(), n);
        if (count == chunk_records) {
          close_chunk();
          open_chunk();
        }
        if (load(&header->stop)) {
          if (head == load(&header->tail))
            break;
          if (!n) {
            store(&header->status, EPROTO);
            break;
          }
        }
        if (!n)
          std::this_thread::sleep_for(std::chrono::microseconds(25));
      }
      if (!load(&header->status))
        close_chunk();
    } catch (...) {
      if (!load(&header->status))
        store(&header->status, EIO);
    }
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }
};
extern "C" {
void stream_store_u32_release(uint32_t *word, uint32_t value) {
  __atomic_store_n(word, value, __ATOMIC_RELEASE);
}
void stream_abort(void *memory, int error) {
  store(&static_cast<Header *>(memory)->status, static_cast<uint64_t>(error));
}
int stream_publish(void *memory, const Record *record) {
  auto h = static_cast<Header *>(memory);
  auto slots = reinterpret_cast<Slot *>(h + 1);
  if (load(&h->status))
    return static_cast<int>(load(&h->status));
  uint64_t ticket = __atomic_fetch_add(&h->tail, 1, __ATOMIC_RELAXED);
  auto &slot = slots[ticket % h->capacity];
  bool waited = false;
  while (load(&slot.sequence) != ticket) {
    if (!waited) {
      __atomic_fetch_add(&h->waits, 1, __ATOMIC_RELAXED);
      waited = true;
    }
    if (load(&h->status))
      return static_cast<int>(load(&h->status));
    std::this_thread::yield();
  }
  slot.record = *record;
  store(&slot.sequence, ticket + 1);
  return 0;
}
void *stream_start(void *memory, const char *path, uint64_t chunk_bytes,
                   uint64_t delay_us, uint64_t fail_after) {
  try {
    return new Writer(memory, path, chunk_bytes, delay_us, fail_after);
  } catch (...) {
    return nullptr;
  }
}
int stream_finish(void *handle, uint64_t *written, uint64_t *chunks) {
  auto w = static_cast<Writer *>(handle);
  store(&w->header->stop, 1);
  w->worker.join();
  *written = w->written.load();
  *chunks = w->chunks.load();
  int status = static_cast<int>(load(&w->header->status));
  delete w;
  return status;
}
}
