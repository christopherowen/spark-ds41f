/* SPDX-License-Identifier: Apache-2.0
 * Row planning for the Engram disk reader, from b12x's PLE reader
 * (local-inference-lab/b12x b12x/loader/_row_plan.h, Apache-2.0): immutable source
 * registration and allocation-free planning of 4 KiB-aligned direct reads. Unlike
 * b12x's, it leaves the slots of other ranks' rows untouched: a rank owns the rows in
 * [tp_start, tp_end) of the hash columns in its mask (lookup i is column i % 24). */

#ifndef B12X_ROW_PLAN_H
#define B12X_ROW_PLAN_H
#include <fcntl.h>
#include <time.h>
#define PLE_BLOCK 4096u
#define PLE_READ_MAX 65536u
#define PLE_COLUMNS 24u
#define PLE_ALL_COLUMNS ((1u << PLE_COLUMNS) - 1u)
#define PLE_PACKED_ROW_BYTES 264u
#define PLE_PACKED_ROWS_PER_PAGE 15u
#define PLE_CACHE_WAYS 4u
#define PLE_SPLIT_TAGS 1
#define PLE_SOA_CACHE 1
#define PLE_DEFAULT_CACHE_ROWS 65536
#define PLE_CACHE_MAX_TOKENS 64u
#define PLE_CACHE_MAX_LOOKUPS (PLE_CACHE_MAX_TOKENS * PLE_COLUMNS)

typedef struct {
    uint8_t clock;
} ple_cache_set_t;

typedef struct {
    int fd;
    dev_t device;
    ino_t inode;
    int64_t bytes;
} ple_file_t;

typedef struct {
    int64_t shard;
    int64_t offset;
    size_t file;
    bool scale;
} ple_source_t;

typedef struct {
    size_t file;
    int64_t offset;
    unsigned length;
    uint32_t lookup;
    size_t destination;
    bool scale;
} ple_fragment_t;

typedef struct {
    size_t file, begin, end;
    int64_t offset;
    unsigned length, expected;
    const char *saved_buffer;
} ple_job_t;

typedef struct {
    int64_t shard_rows, padded_rows, tp_start, tp_end;
    uint32_t columns;
    bool packed, has_col_map;
    int64_t col_start[PLE_COLUMNS], col_end[PLE_COLUMNS], col_delta[PLE_COLUMNS];
    size_t weight_bytes, scale_bytes, max_lookups, capacity;
    ple_fragment_t *fragments;
    ple_job_t *jobs;
    size_t fragment_count, job_count;
    ple_file_t *files;
    ple_source_t *sources;
    size_t file_count, source_count;
    failure_t failure;
    uint64_t lookups, requested_bytes, read_bytes, read_calls;
    uint64_t unique_blocks, coalesced_reads, submit_calls;
    double execution_seconds, planning_seconds;
    ple_cache_set_t *cache;
    int64_t *cache_ids;
    char *cache_weights, *cache_scales;
    size_t cache_sets, cache_set_mask;
    uint64_t cache_populated;
    uint32_t *miss_lookups;
    int64_t *miss_ids;
    size_t miss_count;
    char *miss_weights, *miss_scales;
    const char *miss_src_weights, *miss_src_scales;
    uint64_t cache_hits, cache_misses;
} ple_plan_t;

static inline int ple_source_compare(const void *left, const void *right) {
    const ple_source_t *a = left, *b = right;
    if (a->shard != b->shard) return a->shard < b->shard ? -1 : 1;
    return (int)a->scale - (int)b->scale;
}

static inline void ple_plan_release(ple_plan_t *reader) {
    for (size_t i = 0; i < reader->file_count; i++) close(reader->files[i].fd);
    free(reader->files);
    free(reader->sources);
    free(reader->fragments);
    free(reader->jobs);
    free(reader->cache);
    free(reader->cache_ids);
    free(reader->miss_lookups);
    free(reader->miss_ids);
    free(reader->miss_weights);
    free(reader->miss_scales);
}

static inline int ple_plan_init(ple_plan_t *reader, int64_t shard_rows,
        int64_t padded_rows, int64_t tp_start, int64_t tp_end, uint32_t columns,
        int64_t weight_bytes, int64_t scale_bytes, int64_t max_lookups,
        int64_t cache_rows) {
    if (shard_rows <= 0 || padded_rows <= 0 || tp_start < 0 || tp_end < tp_start ||
        !columns || (columns & ~PLE_ALL_COLUMNS) || cache_rows < 0 ||
        tp_end > padded_rows || weight_bytes <= 0 || scale_bytes < 0 || max_lookups <= 0 ||
        weight_bytes > PY_SSIZE_T_MAX / max_lookups ||
        scale_bytes > PY_SSIZE_T_MAX / max_lookups || max_lookups > PY_SSIZE_T_MAX / 8)
        return EINVAL;
    size_t per_row = (size_t)weight_bytes / PLE_BLOCK + 2;
    if (scale_bytes) per_row += (size_t)scale_bytes / PLE_BLOCK + 2;
    if ((size_t)max_lookups > SIZE_MAX / per_row ||
        (size_t)max_lookups * per_row > SIZE_MAX / sizeof(ple_fragment_t) ||
        (size_t)max_lookups * per_row > SIZE_MAX / sizeof(ple_job_t)) return EOVERFLOW;
    reader->shard_rows = shard_rows;
    reader->padded_rows = padded_rows;
    reader->tp_start = tp_start;
    reader->tp_end = tp_end;
    reader->columns = columns;
    reader->weight_bytes = weight_bytes;
    reader->scale_bytes = scale_bytes;
    reader->max_lookups = max_lookups;
    reader->capacity = (size_t)max_lookups * per_row;
    reader->fragments = calloc(reader->capacity, sizeof(*reader->fragments));
    reader->jobs = calloc(reader->capacity, sizeof(*reader->jobs));
    if (!reader->fragments || !reader->jobs) return ENOMEM;
    if (cache_rows > 0) {
        size_t min_sets = ((size_t)cache_rows + PLE_CACHE_WAYS - 1) / PLE_CACHE_WAYS;
        size_t sets = 1;
        while (sets < min_sets) {
            if (sets > SIZE_MAX / 2) return EOVERFLOW;
            sets <<= 1;
        }
        size_t miss_cap = (size_t)max_lookups > PLE_CACHE_MAX_LOOKUPS
                              ? (size_t)max_lookups
                              : (size_t)PLE_CACHE_MAX_LOOKUPS;
        reader->cache_sets = sets;
        reader->cache_set_mask = sets - 1;
        reader->cache = calloc(sets, sizeof(*reader->cache));
        if (sets > SIZE_MAX / (PLE_CACHE_WAYS * 272u)) return EOVERFLOW;
        if (posix_memalign((void **)&reader->cache_ids, 64, sets * PLE_CACHE_WAYS * 272u))
            reader->cache_ids = NULL;
        else {
            reader->cache_weights = (char *)(reader->cache_ids + sets * PLE_CACHE_WAYS);
            reader->cache_scales = reader->cache_weights + sets * PLE_CACHE_WAYS * 256u;
        }
        reader->miss_lookups = malloc(miss_cap * sizeof(*reader->miss_lookups));
        reader->miss_ids = malloc(miss_cap * sizeof(*reader->miss_ids));
        reader->miss_weights = malloc((size_t)PLE_CACHE_MAX_LOOKUPS * 256u);
        reader->miss_scales = malloc((size_t)PLE_CACHE_MAX_LOOKUPS * 8u);
        if (!reader->cache || !reader->cache_ids || !reader->miss_lookups || !reader->miss_ids ||
            !reader->miss_weights || !reader->miss_scales)
            return ENOMEM;
        for (size_t s = 0; s < sets; s++) {
            for (unsigned w = 0; w < PLE_CACHE_WAYS; w++) {
                reader->cache_ids[s * PLE_CACHE_WAYS + w] = -1;
            }
        }
    }
    return 0;
}

static inline int64_t ple_packed_span_bytes(int64_t rows) {
    if (rows <= 0) return 0;
    int64_t last = rows - 1;
    return (last / (int64_t)PLE_PACKED_ROWS_PER_PAGE) * (int64_t)PLE_BLOCK +
           (last % (int64_t)PLE_PACKED_ROWS_PER_PAGE + 1) * (int64_t)PLE_PACKED_ROW_BYTES;
}

static inline int ple_add_source_rows(ple_plan_t *reader, int64_t shard, const char *path,
        int64_t offset, bool scale, int64_t expected_rows,
        int (*register_files)(ple_plan_t *, void *), void *owner) {
    if (shard < 0 || shard > (reader->padded_rows - 1) / reader->shard_rows || offset < 0 ||
        (scale && !reader->scale_bytes)) return EINVAL;
    int64_t first = shard * reader->shard_rows;
    int64_t rows = expected_rows >= 0 ? expected_rows : (reader->padded_rows - first);
    if (expected_rows < 0 && rows > reader->shard_rows) rows = reader->shard_rows;
    int64_t span_bytes = 0;
    if (reader->packed) {
        if (rows > 0 && (rows - 1) / (int64_t)PLE_PACKED_ROWS_PER_PAGE >
                            (INT64_MAX - offset - (int64_t)PLE_BLOCK) / (int64_t)PLE_BLOCK)
            return EOVERFLOW;
        span_bytes = ple_packed_span_bytes(rows);
    } else {
        size_t row_bytes = scale ? reader->scale_bytes : reader->weight_bytes;
        if (rows > (INT64_MAX - offset) / (int64_t)row_bytes) return EOVERFLOW;
        span_bytes = rows * (int64_t)row_bytes;
    }
    if (expected_rows < 0 && (first >= reader->tp_end || first + rows <= reader->tp_start)) return 0;
    failure_t failure = {{0}};
    ple_source_t key = {.shard = shard, .scale = scale};
    if (reader->source_count && bsearch(&key, reader->sources, reader->source_count,
                                        sizeof(key), ple_source_compare)) {
        snprintf(failure.message, sizeof(failure.message), "PLE source shard/plane is already registered");
    } else {
        /* Reject FIFOs via fstat without blocking waiting for a writer. */
        int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_DIRECT);
        struct stat status;
        if (fd < 0) system_error(&failure, "open PLE source");
        else if (fstat(fd, &status) != 0) system_error(&failure, "fstat PLE source");
        else if (!S_ISREG(status.st_mode) || offset + span_bytes > status.st_size)
            snprintf(failure.message, sizeof(failure.message), "PLE source range exceeds a regular file's size");
        else {
            size_t file = 0;
            while (file < reader->file_count &&
                   (reader->files[file].device != status.st_dev || reader->files[file].inode != status.st_ino)) file++;
            ple_source_t *sources = realloc(reader->sources, (reader->source_count + 1) * sizeof(*sources));
            if (!sources) snprintf(failure.message, sizeof(failure.message), "could not allocate PLE source descriptor");
            else {
                reader->sources = sources;
                if (file == reader->file_count) {
                    ple_file_t *files = realloc(reader->files, (file + 1) * sizeof(*files));
                    if (!files) snprintf(failure.message, sizeof(failure.message), "could not allocate PLE file descriptor");
                    else {
                        reader->files = files;
                        reader->files[file] = (ple_file_t){fd, status.st_dev, status.st_ino, status.st_size};
                        reader->file_count++;
                        fd = -1;
                        int error = register_files ? register_files(reader, owner) : 0;
                        if (error < 0) {
                            snprintf(failure.message, sizeof(failure.message),
                                     "row file registration failed: %s", strerror(-error));
                        }
                    }
                } else if (reader->files[file].bytes != status.st_size) {
                    snprintf(failure.message, sizeof(failure.message), "PLE source file size changed during registration");
                }
                if (!failure.message[0]) {
                    reader->sources[reader->source_count++] = (ple_source_t){shard, offset, file, scale != 0};
                    qsort(reader->sources, reader->source_count, sizeof(*sources), ple_source_compare);
                    const char *drop_env = getenv("SPARK_DS41F_ENGRAM_DROP_PAGE_CACHE");
                    if (!drop_env || strcmp(drop_env, "0") != 0) {
                        int target_fd = reader->files[file].fd;
                        int64_t range_start = (offset / (int64_t)PLE_BLOCK) * (int64_t)PLE_BLOCK;
                        int64_t range_end = offset + span_bytes;
                        (void)posix_fadvise(target_fd, range_start, range_end - range_start, POSIX_FADV_DONTNEED);
                    }
                }
            }
        }
        if (fd >= 0) close(fd);
    }
    if (failure.message[0] && !reader->failure.message[0]) reader->failure = failure;
    return failure.message[0] ? EIO : 0;
}

static inline int ple_add_source(ple_plan_t *reader, int64_t shard, const char *path,
        int64_t offset, bool scale, int (*register_files)(ple_plan_t *, void *), void *owner) {
    if (reader->packed) return EINVAL;
    return ple_add_source_rows(reader, shard, path, offset, scale, -1, register_files, owner);
}

static inline int ple_add_packed_source(ple_plan_t *reader, int64_t shard, const char *path,
        int64_t offset, const int64_t *col_start, const int64_t *col_end, const int64_t *col_delta,
        int (*register_files)(ple_plan_t *, void *), void *owner) {
    if (reader->source_count && !reader->packed) return EINVAL;
    int64_t expected_rows = -1;
    if (col_start || col_end || col_delta) {
        if (!col_start || !col_end || !col_delta || shard != 0) return EINVAL;
        expected_rows = 0;
        for (unsigned c = 0; c < PLE_COLUMNS; c++) {
            if (!((reader->columns >> c) & 1u)) continue;
            if (col_start[c] < 0 || col_end[c] < col_start[c] || col_end[c] > reader->padded_rows ||
                col_start[c] < col_delta[c])
                return EINVAL;
            int64_t phys_end = col_end[c] - col_delta[c];
            if (phys_end > expected_rows) expected_rows = phys_end;
        }
        memcpy(reader->col_start, col_start, sizeof(reader->col_start));
        memcpy(reader->col_end, col_end, sizeof(reader->col_end));
        memcpy(reader->col_delta, col_delta, sizeof(reader->col_delta));
        reader->has_col_map = true;
    }
    reader->packed = true;
    reader->weight_bytes = PLE_PACKED_ROW_BYTES;
    reader->scale_bytes = 0;
    return ple_add_source_rows(reader, shard, path, offset, false, expected_rows, register_files, owner);
}

static inline int ple_fragment_compare(const void *left, const void *right) {
    const ple_fragment_t *a = left, *b = right;
    if (a->file != b->file) return a->file < b->file ? -1 : 1;
    return a->offset < b->offset ? -1 : a->offset > b->offset;
}

/* libc qsort may allocate a merge buffer on every run. Heap-sort in place to
 * keep all batch metadata persistent, including for duplicate-heavy batches. */
static inline void ple_sift(ple_fragment_t *items, size_t root, size_t count) {
    ple_fragment_t value = items[root];
    while (root < count / 2) {
        size_t child = root * 2 + 1;
        if (child + 1 < count && ple_fragment_compare(&items[child], &items[child + 1]) < 0)
            child++;
        if (ple_fragment_compare(&value, &items[child]) >= 0) break;
        items[root] = items[child];
        root = child;
    }
    items[root] = value;
}

static inline void ple_sort(ple_plan_t *reader) {
    ple_fragment_t *items = reader->fragments;
    size_t count = reader->fragment_count;
    if (count < 1024) {
        for (size_t i = count / 2; i > 0; i--) ple_sift(items, i - 1, count);
        for (size_t end = count; end > 1; end--) {
            ple_fragment_t value = items[end - 1];
            items[end - 1] = items[0];
            items[0] = value;
            ple_sift(items, 0, end - 1);
        }
        return;
    }
    /* Jobs are built only after sorting. Reuse that fixed-capacity allocation
     * as radix scratch instead of allocating another prefill-sized buffer. */
    _Static_assert(sizeof(ple_job_t) >= sizeof(ple_fragment_t),
                   "job allocation must cover fragment sort scratch");
    ple_fragment_t *scratch = (ple_fragment_t *)reader->jobs;
    ple_fragment_t *source = items, *target = scratch;
    uint64_t offset_variation = 0, file_variation = 0;
    for (size_t i = 1; i < count; i++) {
        offset_variation |= (uint64_t)(items[i].offset ^ items[0].offset);
        file_variation |= (uint64_t)(items[i].file ^ items[0].file);
    }
    /* Stable LSD passes sort offset first, then the primary file key. Skip
     * constant bytes, including the file key for a single-container table. */
    for (int file = 0; file < 2; file++) {
        uint64_t variation = file ? file_variation : offset_variation;
        for (unsigned shift = 0; shift < 64; shift += 8) {
            if (((variation >> shift) & 255) == 0) continue;
            size_t positions[256] = {0};
            for (size_t i = 0; i < count; i++) {
                uint64_t key = file ? (uint64_t)source[i].file : (uint64_t)source[i].offset;
                positions[(key >> shift) & 255]++;
            }
            size_t prefix = 0;
            for (unsigned i = 0; i < 256; i++) {
                size_t size = positions[i];
                positions[i] = prefix;
                prefix += size;
            }
            for (size_t i = 0; i < count; i++) {
                uint64_t key = file ? (uint64_t)source[i].file : (uint64_t)source[i].offset;
                target[positions[(key >> shift) & 255]++] = source[i];
            }
            ple_fragment_t *swap = source;
            source = target;
            target = swap;
        }
    }
    if (source != items) memcpy(items, source, count * sizeof(*items));
}

static inline size_t ple_cache_set_index(int64_t id, size_t mask) {
    uint64_t h = (uint64_t)id * 0x9E3779B97F4A7C15ULL;
    return (size_t)((h ^ (h >> 32)) & mask);
}

static inline void ple_scatter_packed_fragment(const ple_fragment_t *fragment, const char *src,
        char *dst_weights, char *dst_scales) {
    size_t lookup = fragment->lookup;
    if (fragment->length == PLE_PACKED_ROW_BYTES) {
        memcpy(dst_weights + lookup * 256u, src, 256u);
        memcpy(dst_scales + lookup * 8u, src + 256u, 8u);
        return;
    }
    size_t in_row = fragment->destination - lookup * PLE_PACKED_ROW_BYTES;
    size_t end_in_row = in_row + fragment->length;
    if (in_row < 256u) {
        size_t w_len = (end_in_row <= 256u ? end_in_row : 256u) - in_row;
        memcpy(dst_weights + lookup * 256u + in_row, src, w_len);
    }
    if (end_in_row > 256u) {
        size_t s_start = in_row > 256u ? in_row - 256u : 0u;
        size_t s_len = (end_in_row - 256u) - s_start;
        size_t src_off = in_row < 256u ? 256u - in_row : 0u;
        memcpy(dst_scales + lookup * 8u + s_start, src + src_off, s_len);
    }
}

static inline void ple_stash_job(ple_plan_t *reader, size_t index) {
    ple_job_t *job = &reader->jobs[index];
    const char *buffer = job->saved_buffer;
    if (!buffer) return;
    if (!reader->packed) {
        for (size_t i = job->begin; i < job->end; i++) {
            ple_fragment_t *fragment = &reader->fragments[i];
            memcpy((fragment->scale ? reader->miss_scales : reader->miss_weights) + fragment->destination,
                   buffer + (fragment->offset - job->offset), fragment->length);
        }
    } else {
        for (size_t i = job->begin; i < job->end; i++) {
            ple_fragment_t *fragment = &reader->fragments[i];
            ple_scatter_packed_fragment(fragment, buffer + (fragment->offset - job->offset),
                                        reader->miss_weights, reader->miss_scales);
        }
    }
    job->saved_buffer = NULL;
}

static inline void ple_cache_insert(ple_plan_t *reader, int64_t id,
        const char *weight_src, const char *scale_src) {
    size_t set_idx = ple_cache_set_index(id, reader->cache_set_mask);
    ple_cache_set_t *set = &reader->cache[set_idx];
    for (unsigned w = 0; w < PLE_CACHE_WAYS; w++) {
        if (reader->cache_ids[set_idx * PLE_CACHE_WAYS + w] == id) return;
    }
    unsigned w = set->clock & (PLE_CACHE_WAYS - 1);
    set->clock = (uint8_t)((w + 1) & (PLE_CACHE_WAYS - 1));
    reader->cache_ids[set_idx * PLE_CACHE_WAYS + w] = id;
    memcpy(reader->cache_weights + (set_idx * PLE_CACHE_WAYS + w) * 256u, weight_src, 256u);
    memcpy(reader->cache_scales + (set_idx * PLE_CACHE_WAYS + w) * 8u, scale_src, 8u);
    reader->cache_populated++;
}

static inline void ple_cache_commit(ple_plan_t *reader) {
    if (!reader->miss_count) return;
    const char *w_base = reader->miss_src_weights;
    const char *s_base = reader->miss_src_scales;
    if (!w_base) {
        for (size_t j = 0; j < reader->job_count; j++) {
            ple_stash_job(reader, j);
        }
        w_base = reader->miss_weights;
        s_base = reader->miss_scales;
    }
    for (size_t k = 0; k < reader->miss_count; k++) {
        uint32_t lookup = reader->miss_lookups[k];
        int64_t id = reader->miss_ids[k];
        ple_cache_insert(reader, id,
                         w_base + (size_t)lookup * 256u,
                         s_base + (size_t)lookup * 8u);
    }
    reader->miss_count = 0;
}

static inline bool ple_plan(ple_plan_t *reader, const char *ids, size_t count,
        char *dst_weights, char *dst_scales, bool dst_is_cached) {
    reader->fragment_count = reader->job_count = 0;
    reader->miss_count = 0;
    bool use_cache = reader->cache_sets > 0 && (count <= PLE_CACHE_MAX_LOOKUPS || dst_is_cached);
    bool check_cache = reader->cache_sets > 0 && reader->cache_populated > 0 &&
                       (count <= 96u * PLE_COLUMNS || dst_is_cached);
    bool fill_cache = use_cache;
    reader->miss_src_weights = dst_is_cached ? dst_weights : NULL;
    reader->miss_src_scales = dst_is_cached ? dst_scales : NULL;
    for (size_t i = 0; i < count; i++) {
        int64_t id;
        memcpy(&id, ids + i * sizeof(id), sizeof(id));
        unsigned col = (unsigned)(i % PLE_COLUMNS);
        /* Another rank's row (or a dead lookup): its slot is not read; the gather writes
         * zeros for it without loading the slot. */
        if (id < reader->tp_start || id >= reader->tp_end ||
            !((reader->columns >> col) & 1u))
            continue;
        int64_t phys_id = id;
        if (reader->has_col_map) {
            if (id < reader->col_start[col] || id >= reader->col_end[col]) {
                snprintf(reader->failure.message, sizeof(reader->failure.message),
                         "Engram row %lld is outside hash column %u range [%lld, %lld)",
                         (long long)id, col, (long long)reader->col_start[col],
                         (long long)reader->col_end[col]);
                return false;
            }
            phys_id = id - reader->col_delta[col];
        }
        if (check_cache) {
            size_t set_idx = ple_cache_set_index(id, reader->cache_set_mask);
            int hit_way = -1;
            for (unsigned w = 0; w < PLE_CACHE_WAYS; w++) {
                if (reader->cache_ids[set_idx * PLE_CACHE_WAYS + w] == id) {
                    hit_way = (int)w;
                    break;
                }
            }
            if (hit_way >= 0) {
                memcpy(dst_weights + i * 256u, reader->cache_weights + (set_idx * PLE_CACHE_WAYS + hit_way) * 256u, 256u);
                memcpy(dst_scales + i * 8u, reader->cache_scales + (set_idx * PLE_CACHE_WAYS + hit_way) * 8u, 8u);
                reader->cache_hits++;
                continue;
            }
        }
        if (reader->cache_sets > 0) {
            reader->cache_misses++;
            if (fill_cache) {
                size_t m = reader->miss_count++;
                reader->miss_lookups[m] = (uint32_t)i;
                reader->miss_ids[m] = id;
            }
        }
        for (int plane = 0; plane < (reader->scale_bytes ? 2 : 1); plane++) {
            ple_source_t key = {.shard = phys_id / reader->shard_rows, .scale = plane != 0};
            ple_source_t *source = reader->source_count ? bsearch(&key, reader->sources,
                reader->source_count, sizeof(key), ple_source_compare) : NULL;
            if (!source) {
                snprintf(reader->failure.message, sizeof(reader->failure.message),
                         "missing PLE %s source for shard %lld", plane ? "scale" : "weight", (long long)key.shard);
                return false;
            }
            size_t bytes = plane ? reader->scale_bytes : reader->weight_bytes;
            size_t destination = i * bytes;
            int64_t rel_row = phys_id % reader->shard_rows;
            int64_t offset = reader->packed
                ? source->offset + (rel_row / (int64_t)PLE_PACKED_ROWS_PER_PAGE) * (int64_t)PLE_BLOCK +
                      (rel_row % (int64_t)PLE_PACKED_ROWS_PER_PAGE) * (int64_t)PLE_PACKED_ROW_BYTES
                : source->offset + rel_row * (int64_t)bytes;
            reader->requested_bytes += bytes;
            while (bytes) {
                unsigned length = PLE_BLOCK - (uint64_t)offset % PLE_BLOCK;
                if (bytes < length) length = (unsigned)bytes;
                reader->fragments[reader->fragment_count++] =
                    (ple_fragment_t){source->file, offset, length, (uint32_t)i, destination, plane != 0};
                offset += length;
                destination += length;
                bytes -= length;
            }
        }
    }
    ple_sort(reader);
    size_t cursor = 0;
    while (cursor < reader->fragment_count) {
        size_t begin = cursor;
        ple_fragment_t *fragment = &reader->fragments[cursor];
        size_t file = fragment->file;
        int64_t first = fragment->offset & ~(int64_t)(PLE_BLOCK - 1);
        int64_t last = first;
        reader->unique_blocks++;
        cursor++;
        while (cursor < reader->fragment_count) {
            fragment = &reader->fragments[cursor];
            int64_t block = fragment->offset & ~(int64_t)(PLE_BLOCK - 1);
            if (fragment->file != file || (block != last &&
                (block - last != PLE_BLOCK || block - first >= PLE_READ_MAX))) break;
            if (block != last) { reader->unique_blocks++; last = block; }
            cursor++;
        }
        unsigned aligned = (unsigned)(last - first) + PLE_BLOCK;
        int64_t available = reader->files[file].bytes - first;
        unsigned expected = available < aligned ? (unsigned)available : aligned;
        reader->jobs[reader->job_count++] = (ple_job_t){file, begin, cursor, first, aligned, expected, NULL};
        if (last != first) reader->coalesced_reads++;
    }
    return true;
}


#endif
