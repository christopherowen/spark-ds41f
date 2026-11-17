/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Role-major O_DIRECT + io_uring (IORING_OP_READ_FIXED) reader for routed MoE
 * expert slots on unified-memory SM121 (GB10).
 *
 * Each expert's 4 KiB-aligned record in the layer pack file is laid out in two
 * contiguous role slabs:
 *   1. Role 0 (w13 / gate_up_proj):
 *      - w13_tiles (multiple of 8 KiB = 2 x 4 KiB pages)
 *      - w13_meta  (codes13 + optional tail13 + base13, padded to 4 KiB)
 *   2. Role 1 (w2 / down_proj):
 *      - w2_tiles  (multiple of 8 KiB = 2 x 4 KiB pages)
 *      - w2_meta   (codes2 + optional tail2 + base2, padded to 4 KiB)
 *
 * Dynamic slot regions of w13_tiles and w2_tiles (allocated via cudaHostAlloc /
 * pin_memory and mapped zero-copy into the GPU address space) are registered
 * with io_uring_register_buffers as per-slot fixed buffers alongside a 4 KiB-
 * aligned metadata staging buffer. On a slot miss (or cross-layer router
 * lookahead prefetch), submit_reads() queues all w13 SQEs ahead of all w2 SQEs
 * in one io_uring_submit() call. wait_role(ROLE_W13) reaps until all w13 reads
 * finish so the GPU can launch gate_up_proj and SwiGLU immediately while the
 * NVMe link transfers w2 in parallel; wait_role(ROLE_W2) reaps the remaining
 * w2 CQEs before down_proj launches.
 */
#define _GNU_SOURCE
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define READER_CAPSULE "vllm.expert_slots.reader"
#define ABI_VERSION 1
#define RING_DEPTH 1024u
#define PACK_HEADER_BYTES 4096u

enum {
    KIND_W13_TILES = 1u,
    KIND_W13_META = 2u,
    KIND_W2_TILES = 3u,
    KIND_W2_META = 4u
};

enum {
    ROLE_W13 = 0,
    ROLE_W2 = 1,
    ROLE_ALL = 2
};

typedef struct {
    int fd;
    bool o_direct;
    bool ring_ready;
    bool buffers_registered;
    bool files_registered;
    struct io_uring ring;
    struct iovec *iovecs;
    unsigned iovec_count;

    unsigned num_experts;
    unsigned static_slots;
    unsigned dynamic_slots;

    size_t w13_tiles_slot_bytes;
    size_t w13_codes_slot_bytes;
    size_t w13_tails_slot_bytes;
    size_t w13_base_slot_bytes;
    size_t w13_meta_aligned_bytes;
    size_t w13_role_bytes;

    size_t w2_tiles_slot_bytes;
    size_t w2_codes_slot_bytes;
    size_t w2_tails_slot_bytes;
    size_t w2_base_slot_bytes;
    size_t w2_meta_aligned_bytes;
    size_t w2_role_bytes;

    size_t expert_stride;
    size_t meta_slot_stride;
    size_t header_bytes;

    char *w13_tiles_base;
    char *w13_codes_base;
    char *w13_tails_base;
    char *w13_base_base;

    char *w2_tiles_base;
    char *w2_codes_base;
    char *w2_tails_base;
    char *w2_base_base;

    char *meta_staging;

    unsigned pending_w13;
    unsigned pending_w2;

    uint64_t total_bytes_read;
    uint64_t w13_bytes_read;
    uint64_t w2_bytes_read;
    uint64_t w13_wait_ns;
    uint64_t w2_wait_ns;
    uint64_t submissions;
    uint64_t experts_loaded;
    uint64_t lba_ordered_submissions;
    uint64_t contiguous_slot_pairs;
} expert_slot_reader_t;

typedef struct {
    int32_t expert;
    int32_t dyn_slot;
} expert_slot_req_t;

static int cmp_expert_slot_req(const void *a, const void *b) {
    const expert_slot_req_t *ra = (const expert_slot_req_t *)a;
    const expert_slot_req_t *rb = (const expert_slot_req_t *)b;
    if (ra->expert != rb->expert) {
        return (ra->expert < rb->expert) ? -1 : 1;
    }
    if (ra->dyn_slot != rb->dyn_slot) {
        return (ra->dyn_slot < rb->dyn_slot) ? -1 : 1;
    }
    return 0;
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void scatter_w13_meta(expert_slot_reader_t *r, unsigned d) {
    const char *src = r->meta_staging + (size_t)d * r->meta_slot_stride;
    memcpy(r->w13_codes_base + (size_t)d * r->w13_codes_slot_bytes, src, r->w13_codes_slot_bytes);
    src += r->w13_codes_slot_bytes;
    if (r->w13_tails_base != NULL && r->w13_tails_slot_bytes > 0) {
        memcpy(r->w13_tails_base + (size_t)d * r->w13_tails_slot_bytes, src, r->w13_tails_slot_bytes);
        src += r->w13_tails_slot_bytes;
    }
    memcpy(r->w13_base_base + (size_t)d * r->w13_base_slot_bytes, src, r->w13_base_slot_bytes);
}

static void scatter_w2_meta(expert_slot_reader_t *r, unsigned d) {
    const char *src =
        r->meta_staging + (size_t)d * r->meta_slot_stride + r->w13_meta_aligned_bytes;
    memcpy(r->w2_codes_base + (size_t)d * r->w2_codes_slot_bytes, src, r->w2_codes_slot_bytes);
    src += r->w2_codes_slot_bytes;
    if (r->w2_tails_base != NULL && r->w2_tails_slot_bytes > 0) {
        memcpy(r->w2_tails_base + (size_t)d * r->w2_tails_slot_bytes, src, r->w2_tails_slot_bytes);
        src += r->w2_tails_slot_bytes;
    }
    memcpy(r->w2_base_base + (size_t)d * r->w2_base_slot_bytes, src, r->w2_base_slot_bytes);
}

static int reap_one_cqe(expert_slot_reader_t *r, char *err_buf, size_t err_cap) {
    struct io_uring_cqe *cqe = NULL;
    int ret = io_uring_wait_cqe(&r->ring, &cqe);
    while (ret == -EINTR) {
        ret = io_uring_wait_cqe(&r->ring, &cqe);
    }
    if (ret < 0) {
        snprintf(err_buf, err_cap, "io_uring_wait_cqe failed: %s", strerror(-ret));
        return -1;
    }
    int32_t res = cqe->res;
    uint64_t tag = cqe->user_data;
    io_uring_cqe_seen(&r->ring, cqe);

    unsigned kind = (unsigned)(tag >> 32);
    unsigned d = (unsigned)(tag & 0xFFFFFFFFu);
    if (res < 0) {
        snprintf(
            err_buf,
            err_cap,
            "io_uring read failed (kind=%u, slot=%u): %s",
            kind,
            d,
            strerror(-res)
        );
        if (kind == KIND_W13_TILES || kind == KIND_W13_META) {
            if (r->pending_w13 > 0) r->pending_w13--;
        } else {
            if (r->pending_w2 > 0) r->pending_w2--;
        }
        return -1;
    }

    size_t expected = 0;
    if (kind == KIND_W13_TILES) {
        expected = r->w13_tiles_slot_bytes;
        if (r->pending_w13 > 0) r->pending_w13--;
        r->w13_bytes_read += (uint64_t)res;
    } else if (kind == KIND_W13_META) {
        expected = r->w13_meta_aligned_bytes;
        scatter_w13_meta(r, d);
        if (r->pending_w13 > 0) r->pending_w13--;
        r->w13_bytes_read += (uint64_t)res;
    } else if (kind == KIND_W2_TILES) {
        expected = r->w2_tiles_slot_bytes;
        if (r->pending_w2 > 0) r->pending_w2--;
        r->w2_bytes_read += (uint64_t)res;
    } else if (kind == KIND_W2_META) {
        expected = r->w2_meta_aligned_bytes;
        scatter_w2_meta(r, d);
        if (r->pending_w2 > 0) r->pending_w2--;
        r->w2_bytes_read += (uint64_t)res;
    } else {
        snprintf(err_buf, err_cap, "unknown io_uring tag kind=%u", kind);
        return -1;
    }

    r->total_bytes_read += (uint64_t)res;
    if ((size_t)res != expected) {
        snprintf(
            err_buf,
            err_cap,
            "short io_uring read (kind=%u, slot=%u): got %d bytes, expected %zu",
            kind,
            d,
            res,
            expected
        );
        return -1;
    }
    return 0;
}

static void prep_slot_sqe(
    expert_slot_reader_t *r,
    struct io_uring_sqe *sqe,
    void *buf,
    size_t nbytes,
    off_t offset,
    unsigned buf_index,
    uint64_t user_data
) {
    int fd = r->files_registered ? 0 : r->fd;
    if (r->buffers_registered) {
        io_uring_prep_read_fixed(sqe, fd, buf, (unsigned)nbytes, offset, (int)buf_index);
    } else {
        io_uring_prep_read(sqe, fd, buf, (unsigned)nbytes, offset);
    }
    if (r->files_registered) {
        sqe->flags |= IOSQE_FIXED_FILE;
    }
    sqe->user_data = user_data;
}

static void reader_close(expert_slot_reader_t *r) {
    if (r == NULL) {
        return;
    }
    if (r->ring_ready) {
        char err_buf[256];
        while (r->pending_w13 > 0 || r->pending_w2 > 0) {
            if (reap_one_cqe(r, err_buf, sizeof(err_buf)) < 0) {
                break;
            }
        }
        if (r->buffers_registered) {
            io_uring_unregister_buffers(&r->ring);
            r->buffers_registered = false;
        }
        if (r->files_registered) {
            io_uring_unregister_files(&r->ring);
            r->files_registered = false;
        }
        io_uring_queue_exit(&r->ring);
        r->ring_ready = false;
    }
    if (r->fd >= 0) {
        close(r->fd);
        r->fd = -1;
    }
    free(r->iovecs);
    r->iovecs = NULL;
}

static void capsule_destructor(PyObject *capsule) {
    expert_slot_reader_t *r =
        (expert_slot_reader_t *)PyCapsule_GetPointer(capsule, READER_CAPSULE);
    if (r != NULL) {
        reader_close(r);
        free(r);
    }
}

static expert_slot_reader_t *get_reader(PyObject *capsule) {
    expert_slot_reader_t *r =
        (expert_slot_reader_t *)PyCapsule_GetPointer(capsule, READER_CAPSULE);
    if (r == NULL) {
        return NULL;
    }
    if (!r->ring_ready || r->fd < 0) {
        PyErr_SetString(PyExc_RuntimeError, "expert slot reader is closed");
        return NULL;
    }
    return r;
}

static PyObject *py_open_reader(PyObject *self, PyObject *args) {
    (void)self;
    const char *path = NULL;
    unsigned num_experts = 0;
    unsigned static_slots = 0;
    unsigned dynamic_slots = 0;

    unsigned long long w13_tiles_ptr = 0;
    Py_ssize_t w13_tiles_slot_bytes = 0;
    unsigned long long w13_codes_ptr = 0;
    Py_ssize_t w13_codes_slot_bytes = 0;
    unsigned long long w13_tails_ptr = 0;
    Py_ssize_t w13_tails_slot_bytes = 0;
    unsigned long long w13_base_ptr = 0;
    Py_ssize_t w13_base_slot_bytes = 0;
    Py_ssize_t w13_meta_aligned_bytes = 0;

    unsigned long long w2_tiles_ptr = 0;
    Py_ssize_t w2_tiles_slot_bytes = 0;
    unsigned long long w2_codes_ptr = 0;
    Py_ssize_t w2_codes_slot_bytes = 0;
    unsigned long long w2_tails_ptr = 0;
    Py_ssize_t w2_tails_slot_bytes = 0;
    unsigned long long w2_base_ptr = 0;
    Py_ssize_t w2_base_slot_bytes = 0;
    Py_ssize_t w2_meta_aligned_bytes = 0;

    unsigned long long meta_staging_ptr = 0;

    if (!PyArg_ParseTuple(
            args,
            "sIIIKnKnKnKnnKnKnKnKnnK",
            &path,
            &num_experts,
            &static_slots,
            &dynamic_slots,
            &w13_tiles_ptr,
            &w13_tiles_slot_bytes,
            &w13_codes_ptr,
            &w13_codes_slot_bytes,
            &w13_tails_ptr,
            &w13_tails_slot_bytes,
            &w13_base_ptr,
            &w13_base_slot_bytes,
            &w13_meta_aligned_bytes,
            &w2_tiles_ptr,
            &w2_tiles_slot_bytes,
            &w2_codes_ptr,
            &w2_codes_slot_bytes,
            &w2_tails_ptr,
            &w2_tails_slot_bytes,
            &w2_base_ptr,
            &w2_base_slot_bytes,
            &w2_meta_aligned_bytes,
            &meta_staging_ptr
        )) {
        return NULL;
    }

    expert_slot_reader_t *r = (expert_slot_reader_t *)calloc(1, sizeof(*r));
    if (r == NULL) {
        return PyErr_NoMemory();
    }
    r->fd = -1;
    r->num_experts = num_experts;
    r->static_slots = static_slots;
    r->dynamic_slots = dynamic_slots;

    r->w13_tiles_slot_bytes = (size_t)w13_tiles_slot_bytes;
    r->w13_codes_slot_bytes = (size_t)w13_codes_slot_bytes;
    r->w13_tails_slot_bytes = (size_t)w13_tails_slot_bytes;
    r->w13_base_slot_bytes = (size_t)w13_base_slot_bytes;
    r->w13_meta_aligned_bytes = (size_t)w13_meta_aligned_bytes;
    r->w13_role_bytes = r->w13_tiles_slot_bytes + r->w13_meta_aligned_bytes;

    r->w2_tiles_slot_bytes = (size_t)w2_tiles_slot_bytes;
    r->w2_codes_slot_bytes = (size_t)w2_codes_slot_bytes;
    r->w2_tails_slot_bytes = (size_t)w2_tails_slot_bytes;
    r->w2_base_slot_bytes = (size_t)w2_base_slot_bytes;
    r->w2_meta_aligned_bytes = (size_t)w2_meta_aligned_bytes;
    r->w2_role_bytes = r->w2_tiles_slot_bytes + r->w2_meta_aligned_bytes;

    r->expert_stride = r->w13_role_bytes + r->w2_role_bytes;
    r->meta_slot_stride = r->w13_meta_aligned_bytes + r->w2_meta_aligned_bytes;

    r->w13_tiles_base = (char *)(uintptr_t)w13_tiles_ptr;
    r->w13_codes_base = (char *)(uintptr_t)w13_codes_ptr;
    r->w13_tails_base = w13_tails_ptr ? (char *)(uintptr_t)w13_tails_ptr : NULL;
    r->w13_base_base = (char *)(uintptr_t)w13_base_ptr;

    r->w2_tiles_base = (char *)(uintptr_t)w2_tiles_ptr;
    r->w2_codes_base = (char *)(uintptr_t)w2_codes_ptr;
    r->w2_tails_base = w2_tails_ptr ? (char *)(uintptr_t)w2_tails_ptr : NULL;
    r->w2_base_base = (char *)(uintptr_t)w2_base_ptr;

    r->meta_staging = (char *)(uintptr_t)meta_staging_ptr;

    int fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd >= 0) {
        r->fd = fd;
        r->o_direct = true;
    } else if (errno == EINVAL) {
        fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
            free(r);
            return NULL;
        }
        r->fd = fd;
        r->o_direct = false;
    } else {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
        free(r);
        return NULL;
    }

    struct stat st;
    r->header_bytes = 0;
    if (fstat(r->fd, &st) == 0) {
        size_t body_bytes = (size_t)r->num_experts * r->expert_stride;
        if ((size_t)st.st_size == (size_t)PACK_HEADER_BYTES + body_bytes) {
            r->header_bytes = (size_t)PACK_HEADER_BYTES;
        }
    }

    int ret = io_uring_queue_init(RING_DEPTH, &r->ring, 0);
    if (ret < 0) {
        close(r->fd);
        free(r);
        errno = -ret;
        return PyErr_SetFromErrno(PyExc_OSError);
    }
    r->ring_ready = true;

    if (io_uring_register_files(&r->ring, &r->fd, 1) == 0) {
        r->files_registered = true;
    }

    if (dynamic_slots > 0) {
        unsigned iov_count = 2u * dynamic_slots + 1u;
        r->iovecs = (struct iovec *)calloc(iov_count, sizeof(struct iovec));
        if (r->iovecs == NULL) {
            reader_close(r);
            free(r);
            return PyErr_NoMemory();
        }
        r->iovec_count = iov_count;
        for (unsigned d = 0; d < dynamic_slots; d++) {
            r->iovecs[d].iov_base = r->w13_tiles_base + (size_t)d * r->w13_tiles_slot_bytes;
            r->iovecs[d].iov_len = r->w13_tiles_slot_bytes;
            r->iovecs[dynamic_slots + d].iov_base =
                r->w2_tiles_base + (size_t)d * r->w2_tiles_slot_bytes;
            r->iovecs[dynamic_slots + d].iov_len = r->w2_tiles_slot_bytes;
        }
        r->iovecs[2u * dynamic_slots].iov_base = r->meta_staging;
        r->iovecs[2u * dynamic_slots].iov_len = (size_t)dynamic_slots * r->meta_slot_stride;

        if (io_uring_register_buffers(&r->ring, r->iovecs, iov_count) == 0) {
            r->buffers_registered = true;
        }
    }

    PyObject *cap = PyCapsule_New(r, READER_CAPSULE, capsule_destructor);
    if (cap == NULL) {
        reader_close(r);
        free(r);
        return NULL;
    }
    return cap;
}

static PyObject *py_submit_reads(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    Py_buffer experts_buf = {0};
    Py_buffer slots_buf = {0};

    if (!PyArg_ParseTuple(
            args, "Oy*y*", &capsule, &experts_buf, &slots_buf
        )) {
        return NULL;
    }
    expert_slot_reader_t *r = get_reader(capsule);
    if (r == NULL) {
        PyBuffer_Release(&experts_buf);
        PyBuffer_Release(&slots_buf);
        return NULL;
    }

    if (experts_buf.len != slots_buf.len || (experts_buf.len % (Py_ssize_t)sizeof(int32_t)) != 0) {
        PyBuffer_Release(&experts_buf);
        PyBuffer_Release(&slots_buf);
        PyErr_SetString(PyExc_ValueError, "experts and dyn_slots must be equal-length int32 buffers");
        return NULL;
    }

    unsigned count = (unsigned)(experts_buf.len / (Py_ssize_t)sizeof(int32_t));
    const int32_t *experts = (const int32_t *)experts_buf.buf;
    const int32_t *dyn_slots = (const int32_t *)slots_buf.buf;

    expert_slot_req_t *reqs = NULL;
    if (count > 0) {
        reqs = (expert_slot_req_t *)malloc((size_t)count * sizeof(expert_slot_req_t));
        if (reqs == NULL) {
            PyBuffer_Release(&experts_buf);
            PyBuffer_Release(&slots_buf);
            return PyErr_NoMemory();
        }
    }
    for (unsigned i = 0; i < count; i++) {
        if (experts[i] < 0 || (unsigned)experts[i] >= r->num_experts) {
            free(reqs);
            PyBuffer_Release(&experts_buf);
            PyBuffer_Release(&slots_buf);
            PyErr_Format(PyExc_ValueError, "expert id %d out of range 0..%u", experts[i], r->num_experts);
            return NULL;
        }
        if (dyn_slots[i] < 0 || (unsigned)dyn_slots[i] >= r->dynamic_slots) {
            free(reqs);
            PyBuffer_Release(&experts_buf);
            PyBuffer_Release(&slots_buf);
            PyErr_Format(PyExc_ValueError, "dynamic slot %d out of range 0..%u", dyn_slots[i], r->dynamic_slots);
            return NULL;
        }
        reqs[i].expert = experts[i];
        reqs[i].dyn_slot = dyn_slots[i];
    }
    if (count > 1) {
        qsort(reqs, (size_t)count, sizeof(expert_slot_req_t), cmp_expert_slot_req);
    }

    char err_buf[256] = {0};
    int err = 0;
    unsigned D = r->dynamic_slots;

    Py_BEGIN_ALLOW_THREADS
    for (unsigned i = 1; i < count; i++) {
        if (reqs[i].expert == reqs[i - 1].expert + 1 &&
            reqs[i].dyn_slot == reqs[i - 1].dyn_slot + 1) {
            r->contiguous_slot_pairs++;
        }
    }

    /* Phase 1: submit all w13 reads first in ascending LBA order. */
    for (unsigned i = 0; i < count && err == 0; i++) {
        while (r->pending_w13 + r->pending_w2 + 2u > RING_DEPTH) {
            io_uring_submit(&r->ring);
            if (reap_one_cqe(r, err_buf, sizeof(err_buf)) < 0) {
                err = -1;
                break;
            }
        }
        if (err < 0) break;

        unsigned e = (unsigned)reqs[i].expert;
        unsigned d = (unsigned)reqs[i].dyn_slot;
        off_t base_off = (off_t)(r->header_bytes + (size_t)e * r->expert_stride);

        struct io_uring_sqe *sqe_tiles = io_uring_get_sqe(&r->ring);
        if (sqe_tiles == NULL) {
            io_uring_submit(&r->ring);
            sqe_tiles = io_uring_get_sqe(&r->ring);
        }
        prep_slot_sqe(
            r,
            sqe_tiles,
            r->w13_tiles_base + (size_t)d * r->w13_tiles_slot_bytes,
            r->w13_tiles_slot_bytes,
            base_off,
            d,
            ((uint64_t)KIND_W13_TILES << 32) | (uint64_t)d
        );
        r->pending_w13++;

        struct io_uring_sqe *sqe_meta = io_uring_get_sqe(&r->ring);
        if (sqe_meta == NULL) {
            io_uring_submit(&r->ring);
            sqe_meta = io_uring_get_sqe(&r->ring);
        }
        prep_slot_sqe(
            r,
            sqe_meta,
            r->meta_staging + (size_t)d * r->meta_slot_stride,
            r->w13_meta_aligned_bytes,
            base_off + (off_t)r->w13_tiles_slot_bytes,
            2u * D,
            ((uint64_t)KIND_W13_META << 32) | (uint64_t)d
        );
        r->pending_w13++;
    }

    /* Phase 2: submit all w2 reads after w13 reads. */
    for (unsigned i = 0; i < count && err == 0; i++) {
        while (r->pending_w13 + r->pending_w2 + 2u > RING_DEPTH) {
            io_uring_submit(&r->ring);
            if (reap_one_cqe(r, err_buf, sizeof(err_buf)) < 0) {
                err = -1;
                break;
            }
        }
        if (err < 0) break;

        unsigned e = (unsigned)reqs[i].expert;
        unsigned d = (unsigned)reqs[i].dyn_slot;
        off_t w2_off = (off_t)(r->header_bytes + (size_t)e * r->expert_stride + r->w13_role_bytes);

        struct io_uring_sqe *sqe_tiles = io_uring_get_sqe(&r->ring);
        if (sqe_tiles == NULL) {
            io_uring_submit(&r->ring);
            sqe_tiles = io_uring_get_sqe(&r->ring);
        }
        prep_slot_sqe(
            r,
            sqe_tiles,
            r->w2_tiles_base + (size_t)d * r->w2_tiles_slot_bytes,
            r->w2_tiles_slot_bytes,
            w2_off,
            D + d,
            ((uint64_t)KIND_W2_TILES << 32) | (uint64_t)d
        );
        r->pending_w2++;

        struct io_uring_sqe *sqe_meta = io_uring_get_sqe(&r->ring);
        if (sqe_meta == NULL) {
            io_uring_submit(&r->ring);
            sqe_meta = io_uring_get_sqe(&r->ring);
        }
        prep_slot_sqe(
            r,
            sqe_meta,
            r->meta_staging + (size_t)d * r->meta_slot_stride + r->w13_meta_aligned_bytes,
            r->w2_meta_aligned_bytes,
            w2_off + (off_t)r->w2_tiles_slot_bytes,
            2u * D,
            ((uint64_t)KIND_W2_META << 32) | (uint64_t)d
        );
        r->pending_w2++;
    }

    if (err == 0 && count > 0) {
        int sret = io_uring_submit(&r->ring);
        if (sret < 0) {
            snprintf(err_buf, sizeof(err_buf), "io_uring_submit failed: %s", strerror(-sret));
            err = -1;
        } else {
            r->submissions++;
            r->lba_ordered_submissions++;
            r->experts_loaded += count;
        }
    }
    Py_END_ALLOW_THREADS

    free(reqs);
    PyBuffer_Release(&experts_buf);
    PyBuffer_Release(&slots_buf);

    if (err < 0) {
        PyErr_SetString(PyExc_RuntimeError, err_buf);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *py_wait_role(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    int role = ROLE_ALL;

    if (!PyArg_ParseTuple(args, "Oi", &capsule, &role)) {
        return NULL;
    }
    expert_slot_reader_t *r = get_reader(capsule);
    if (r == NULL) {
        return NULL;
    }

    char err_buf[256] = {0};
    int err = 0;

    Py_BEGIN_ALLOW_THREADS
    uint64_t t0 = monotonic_ns();
    if (role == ROLE_W13) {
        while (r->pending_w13 > 0) {
            if (reap_one_cqe(r, err_buf, sizeof(err_buf)) < 0) {
                err = -1;
                break;
            }
        }
        r->w13_wait_ns += monotonic_ns() - t0;
    } else if (role == ROLE_W2) {
        while (r->pending_w2 > 0) {
            if (reap_one_cqe(r, err_buf, sizeof(err_buf)) < 0) {
                err = -1;
                break;
            }
        }
        r->w2_wait_ns += monotonic_ns() - t0;
    } else {
        while (r->pending_w13 > 0 || r->pending_w2 > 0) {
            if (reap_one_cqe(r, err_buf, sizeof(err_buf)) < 0) {
                err = -1;
                break;
            }
        }
        r->w2_wait_ns += monotonic_ns() - t0;
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    Py_END_ALLOW_THREADS

    if (err < 0) {
        PyErr_SetString(PyExc_RuntimeError, err_buf);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *py_stats(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }
    expert_slot_reader_t *r = get_reader(capsule);
    if (r == NULL) {
        return NULL;
    }
    return Py_BuildValue(
        "{s:O,s:O,s:O,s:I,s:I,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K}",
        "o_direct", r->o_direct ? Py_True : Py_False,
        "buffers_registered", r->buffers_registered ? Py_True : Py_False,
        "files_registered", r->files_registered ? Py_True : Py_False,
        "pending_w13", r->pending_w13,
        "pending_w2", r->pending_w2,
        "header_bytes", (unsigned long long)r->header_bytes,
        "total_bytes_read", (unsigned long long)r->total_bytes_read,
        "w13_bytes_read", (unsigned long long)r->w13_bytes_read,
        "w2_bytes_read", (unsigned long long)r->w2_bytes_read,
        "w13_wait_ns", (unsigned long long)r->w13_wait_ns,
        "w2_wait_ns", (unsigned long long)r->w2_wait_ns,
        "submissions", (unsigned long long)r->submissions,
        "experts_loaded", (unsigned long long)r->experts_loaded,
        "lba_ordered_submissions", (unsigned long long)r->lba_ordered_submissions,
        "contiguous_slot_pairs", (unsigned long long)r->contiguous_slot_pairs
    );
}

static PyObject *py_close_reader(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }
    expert_slot_reader_t *r =
        (expert_slot_reader_t *)PyCapsule_GetPointer(capsule, READER_CAPSULE);
    if (r == NULL) {
        return NULL;
    }
    reader_close(r);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"open_reader", py_open_reader, METH_VARARGS, "Open an O_DIRECT io_uring expert slot reader."},
    {"submit_reads", py_submit_reads, METH_VARARGS, "Submit role-major w13 and w2 reads into dynamic slots."},
    {"wait_role", py_wait_role, METH_VARARGS, "Wait for w13 (0), w2 (1), or all (2) pending reads."},
    {"stats", py_stats, METH_VARARGS, "Return reader telemetry and registration flags."},
    {"close_reader", py_close_reader, METH_VARARGS, "Close the reader and unregister io_uring resources."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_expert_slots_reader",
    "Role-major O_DIRECT io_uring reader for routed MoE expert slots.",
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__expert_slots_reader(void) {
    PyObject *m = PyModule_Create(&module_def);
    if (m == NULL) {
        return NULL;
    }
    if (PyModule_AddIntConstant(m, "ABI_VERSION", ABI_VERSION) < 0) return NULL;
    if (PyModule_AddIntConstant(m, "ROLE_W13", ROLE_W13) < 0) return NULL;
    if (PyModule_AddIntConstant(m, "ROLE_W2", ROLE_W2) < 0) return NULL;
    if (PyModule_AddIntConstant(m, "ROLE_ALL", ROLE_ALL) < 0) return NULL;
    return m;
}
