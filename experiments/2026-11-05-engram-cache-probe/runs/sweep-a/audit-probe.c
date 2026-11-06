/* SPDX-License-Identifier: Apache-2.0
 * Offline planner replay: no file reads, io_uring, CUDA or serving timing.
 * The real planner is compiled unmodified from the named source directory.
 */
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
#ifndef POSIX_FADV_DONTNEED
#define POSIX_FADV_DONTNEED 4
static int posix_fadvise(int fd, off_t off, off_t len, int advice) {
    (void)fd; (void)off; (void)len; (void)advice; abort();
}
#endif
#define PY_SSIZE_T_MAX INTPTR_MAX
typedef struct { char message[512]; } failure_t;
static void system_error(failure_t *failure, const char *context) {
    snprintf(failure->message, sizeof(failure->message), "%s: %s", context, strerror(errno));
}
#include "_row_plan.h"

static uint64_t rng = 0x123456789abcdefULL;
static uint64_t random_word(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng;
}
static uint64_t now_ns(void) {
    struct timespec t; assert(clock_gettime(CLOCK_MONOTONIC, &t) == 0);
    return (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec;
}
static unsigned char byte_at(int64_t id, size_t byte, bool scale) {
    return (unsigned char)((uint64_t)id * 131 + byte * 17 + (scale ? 91 : 0));
}
static void fill_and_check(ple_plan_t *p, int64_t *ids, size_t count,
                           char *weights, char *scales) {
    for (size_t i = 0; i < p->fragment_count; i++) {
        ple_fragment_t *f = &p->fragments[i];
        assert(f->file < 2);
        int64_t start = f->scale ? 123 : 4093;
        size_t stride = f->scale ? 8 : 256;
        char *dst = f->scale ? scales : weights;
        for (unsigned j = 0; j < f->length; j++) {
            int64_t pos = f->offset - start + j;
            assert(pos >= 0);
            dst[f->destination + j] = byte_at(pos / stride, pos % stride, f->scale);
        }
    }
    for (size_t i = 0; i < count; i++) {
        bool owned = ((p->columns >> (i % 24)) & 1) && ids[i] >= 0 && ids[i] < p->padded_rows;
        for (size_t j = 0; j < 256; j++)
            assert((unsigned char)weights[i * 256 + j] ==
                   (owned ? byte_at(ids[i], j, false) : 0xa5));
        for (size_t j = 0; j < 8; j++)
            assert((unsigned char)scales[i * 8 + j] ==
                   (owned ? byte_at(ids[i], j, true) : 0xa5));
    }
}
static void warm(ple_plan_t *p, unsigned rows_per_column) {
    char w[256], s[8];
    for (unsigned c = 0; c < 24; c++) {
        if (!((p->columns >> c) & 1)) continue;
        for (unsigned r = 0; r < rows_per_column; r++) {
            int64_t id = (int64_t)c * 16000000 + r;
            for (size_t j = 0; j < 256; j++) w[j] = byte_at(id, j, false);
            for (size_t j = 0; j < 8; j++) s[j] = byte_at(id, j, true);
            ple_cache_insert(p, id, w, s);
        }
    }
}
int main(int argc, char **argv) {
    assert(argc == 8);
    size_t tokens = strtoul(argv[1], NULL, 10);
    unsigned tp = strtoul(argv[2], NULL, 10);
    unsigned hot_pct = strtoul(argv[3], NULL, 10);
    bool populated = atoi(argv[4]) != 0;
    unsigned rounds = strtoul(argv[5], NULL, 10);
    unsigned warm_rows = strtoul(argv[6], NULL, 10);
    unsigned repetitions = strtoul(argv[7], NULL, 10);
    assert(warm_rows > 0 && warm_rows < 16000000 && repetitions > 0);
    assert(tokens > 0 && tokens <= 16384 && (tp == 3 || tp == 4));
    assert(hot_pct <= 100 && rounds > 0);
    size_t count = tokens * 24;
    uint32_t mask = 0;
    for (unsigned c = 1; c < 24; c += tp) mask |= 1u << c;
    ple_plan_t p = {0};
    assert(ple_plan_init(&p, 384000000, 384000000, 0, 384000000,
                         mask, 256, 8, count, 65536) == 0);
    p.file_count = p.source_count = 2;
    p.files = calloc(2, sizeof(*p.files)); p.sources = calloc(2, sizeof(*p.sources));
    assert(p.files && p.sources);
    p.files[0] = (ple_file_t){.fd = -1, .bytes = 4093LL + 384000000LL * 256};
    p.files[1] = (ple_file_t){.fd = -1, .bytes = 123LL + 384000000LL * 8};
    p.sources[0] = (ple_source_t){.offset = 4093, .file = 0};
    p.sources[1] = (ple_source_t){.offset = 123, .file = 1, .scale = true};
    int64_t *ids = malloc(count * sizeof(*ids));
    char *weights = malloc(count * 256), *scales = malloc(count * 8);
    char *cache_before = malloc(p.cache_sets * sizeof(*p.cache));
    assert(ids && weights && scales && cache_before);
    uint64_t warm_start = now_ns();
    if (populated) warm(&p, warm_rows);
    uint64_t warm_elapsed = now_ns() - warm_start;
    memcpy(cache_before, p.cache, p.cache_sets * sizeof(*p.cache));
#if defined(PLE_SPLIT_TAGS)
    size_t extra_bytes = p.cache_sets * PLE_CACHE_WAYS * sizeof(*p.cache_ids);
    char *extra = (char *)p.cache_ids;
#elif defined(PLE_FILTER_TAGS)
    size_t extra_bytes = p.cache_sets * sizeof(*p.cache_filter);
    char *extra = (char *)p.cache_filter;
#else
    size_t extra_bytes = 0;
    char *extra = NULL;
#endif
    char *extra_before = malloc(extra_bytes ? extra_bytes : 1);
    assert(extra_before);
    if (extra_bytes) memcpy(extra_before, extra, extra_bytes);
    uint64_t digest = 1469598103934665603ULL;
    printf("{\"warm_rows\":%u,\"repetitions\":%u,\"extra_cache_bytes\":%zu,\"warm_ns\":%" PRIu64 ",", warm_rows, repetitions, extra_bytes, warm_elapsed);
    printf("\"tokens\":%zu,\"tp\":%u,\"hot_pct\":%u,\"populated\":%s,"
           "\"cache_allocation_bytes\":%zu,\"samples\":[", tokens, tp, hot_pct,
           populated ? "true" : "false", p.cache_sets * sizeof(*p.cache));
    for (unsigned run = 0; run < rounds; run++) {
        for (size_t i = 0; i < count; i++) {
            uint64_t r = random_word();
            ids[i] = (int64_t)(i % 24) * 16000000 +
                     (r % 100 < hot_pct ? (r >> 8) % warm_rows : warm_rows + (r >> 8) % (16000000 - warm_rows));
            if (i % 97 == 0) ids[i] = -1; /* padding, dead, or other unavailable rows */
            digest = (digest ^ (uint64_t)ids[i]) * 1099511628211ULL;
        }
        memset(weights, 0xa5, count * 256); memset(scales, 0xa5, count * 8);
        p.unique_blocks = p.requested_bytes = p.cache_hits = p.cache_misses = 0;
        uint64_t start = now_ns();
        for (unsigned repeat = 0; repeat < repetitions; repeat++) {
            p.unique_blocks = p.requested_bytes = p.cache_hits = p.cache_misses = 0;
            assert(ple_plan(&p, (const char *)ids, count, weights, scales, false));
        }
        uint64_t elapsed = (now_ns() - start) / repetitions;
        uint64_t bytes = 0;
        for (size_t j = 0; j < p.job_count; j++) bytes += p.jobs[j].length;
        fill_and_check(&p, ids, count, weights, scales);
        if (tokens > 64) assert(p.miss_count == 0); /* long steps never admit */
        assert(memcmp(cache_before, p.cache, p.cache_sets * sizeof(*p.cache)) == 0);
        if (extra_bytes) assert(memcmp(extra_before, extra, extra_bytes) == 0);
        printf("%s{\"plan_ns\":%" PRIu64 ",\"planned_jobs\":%zu,\"planned_bytes\":%" PRIu64
               ",\"unique_blocks\":%" PRIu64 ",\"cache_hits\":%" PRIu64
               ",\"cache_misses\":%" PRIu64 "}", run ? "," : "", elapsed,
               p.job_count, bytes, p.unique_blocks, p.cache_hits, p.cache_misses);
    }
    /* Missing source must still fail; cache hits cannot excuse a distinct miss. */
    p.source_count = 0; ids[1] = 16000000 + 9000000;
    assert(!ple_plan(&p, (const char *)ids, count, weights, scales, false));
    assert(strstr(p.failure.message, "missing PLE"));
    struct rusage usage;
    assert(getrusage(RUSAGE_SELF, &usage) == 0);
#ifdef __APPLE__
    uint64_t peak_rss = (uint64_t)usage.ru_maxrss;
#else
    uint64_t peak_rss = (uint64_t)usage.ru_maxrss * 1024u;
#endif
    printf("],\"ids_digest_fnv64\":\"%016" PRIx64 "\",\"byte_checks\":true,"
           "\"cache_unchanged\":true,\"missing_source_rejected\":true,"
           "\"peak_rss_bytes\":%" PRIu64 "}\n", digest, peak_rss);
    free(extra_before); free(ids); free(weights); free(scales); free(cache_before); ple_plan_release(&p);
    return 0;
}
