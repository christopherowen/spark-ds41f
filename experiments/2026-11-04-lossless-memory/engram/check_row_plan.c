/* SPDX-License-Identifier: Apache-2.0
 * Portable metadata/fragment test only; O_DIRECT is unused. No NVMe or GPU timing. */
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
#include <sys/types.h>
#include <unistd.h>
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
#define PY_SSIZE_T_MAX INTPTR_MAX
typedef struct { char message[512]; } failure_t;
static void system_error(failure_t *failure, const char *context) {
    snprintf(failure->message, sizeof(failure->message), "%s: %s", context, strerror(errno));
}
#include "_row_plan.h"
int main(void) {
    ple_plan_t reader = {0};
    assert(ple_plan_init(&reader, 1, 1, 0, 1, 1, 264, 0, 1) == 0);
    int64_t start[24] = {0}, end[24] = {1}, delta[24] = {INT64_MIN};
    int result = ple_add_packed_source(&reader, 0, "/nonexistent", 0,
                                       start, end, delta, NULL, NULL);
    ple_plan_release(&reader);
    if (result != EINVAL) {
        fprintf(stderr, "invalid column delta returned %d instead of EINVAL\n", result);
        return 1;
    }
    assert(reader.failure.message[0] == 0);
    ple_plan_t valid = {0};
    assert(ple_plan_init(&valid, 1025, 2025, 1000, 2025, 1, 264, 0, 1) == 0);
    valid.packed = valid.has_col_map = true;
    valid.col_start[0] = 1000;
    valid.col_end[0] = 2025;
    valid.col_delta[0] = 1000;
    valid.file_count = valid.source_count = 1;
    valid.files = calloc(1, sizeof(*valid.files));
    valid.sources = calloc(1, sizeof(*valid.sources));
    assert(valid.files && valid.sources);
    valid.files[0].fd = -1;
    valid.files[0].bytes = 1025 * 264;
    valid.sources[0] = (ple_source_t){.shard = 0, .offset = 0, .file = 0, .scale = false};
    for (int64_t row = 0; row < 1025; ++row) {
        int64_t id = row + 1000;
        unsigned char output[264] = {0}, covered[264] = {0};
        assert(ple_plan(&valid, (const char *)&id, 1));
        assert(valid.fragment_count >= 1 && valid.fragment_count <= 2);
        for (size_t f = 0; f < valid.fragment_count; ++f) {
            ple_fragment_t fragment = valid.fragments[f];
            assert(fragment.file == 0 && !fragment.scale);
            assert(fragment.offset >= 0 && fragment.offset + fragment.length <= valid.files[0].bytes);
            for (unsigned i = 0; i < fragment.length; ++i) {
                size_t destination = fragment.destination + i;
                assert(destination < sizeof(output) && covered[destination] == 0);
                covered[destination] = 1;
                output[destination] = (unsigned char)(fragment.offset + i);
            }
        }
        for (size_t i = 0; i < sizeof(output); ++i) {
            assert(covered[i] == 1);
            assert(output[i] == (unsigned char)(row * 264 + i));
        }
    }
    ple_plan_release(&valid);
    puts("negative delta rejected; 1025 compacted rows preserve every byte across page boundaries");
    return 0;
}
