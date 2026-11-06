#!/usr/bin/env python3
"""Compare probe bounds and cache metadata layouts on the exact native planner.

This is a CPU-only mechanism screen. It does not issue reads or model CUDA
host-mapped destination memory. Generated headers are disposable; patches and
every command/result are retained. Run only on a bounded remote worker.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time


def replace(text, before, after, count=1):
    assert text.count(before) == count, (before, text.count(before), count)
    return text.replace(before, after)


def bound(source, maximum):
    if maximum == 64:
        return source
    return replace(source, 'bool check_cache = use_cache && reader->cache_populated > 0;',
                   'bool check_cache = reader->cache_sets > 0 && reader->cache_populated > 0 &&\n'
                   f'                       (count <= {maximum}u * PLE_COLUMNS || dst_is_cached);')


def split_tags(source):
    source = replace(source, '#define PLE_CACHE_WAYS 4u', '#define PLE_CACHE_WAYS 4u\n#define PLE_SPLIT_TAGS 1')
    source = replace(source, '    int64_t ids[PLE_CACHE_WAYS];\n', '')
    source = replace(source, '    ple_cache_set_t *cache;', '    ple_cache_set_t *cache;\n    int64_t *cache_ids;')
    source = replace(source, '    free(reader->cache);', '    free(reader->cache);\n    free(reader->cache_ids);')
    source = replace(source, '        reader->cache = calloc(sets, sizeof(*reader->cache));',
                     '        reader->cache = calloc(sets, sizeof(*reader->cache));\n'
                     '        reader->cache_ids = malloc(sets * PLE_CACHE_WAYS * sizeof(*reader->cache_ids));')
    source = replace(source, 'if (!reader->cache || !reader->miss_lookups',
                     'if (!reader->cache || !reader->cache_ids || !reader->miss_lookups')
    source = replace(source, 'reader->cache[s].ids[w] = -1;', 'reader->cache_ids[s * PLE_CACHE_WAYS + w] = -1;')
    source = replace(source, 'set->ids[w]', 'reader->cache_ids[set_idx * PLE_CACHE_WAYS + w]', count=3)
    return source


def filter_tags(source):
    source = replace(source, '#define PLE_CACHE_WAYS 4u', '#define PLE_CACHE_WAYS 4u\n#define PLE_FILTER_TAGS 1')
    source = replace(source, '    ple_cache_set_t *cache;', '    ple_cache_set_t *cache;\n    uint64_t *cache_filter;')
    source = replace(source, '    free(reader->cache);', '    free(reader->cache);\n    free(reader->cache_filter);')
    source = replace(source, '        reader->cache = calloc(sets, sizeof(*reader->cache));',
                     '        reader->cache = calloc(sets, sizeof(*reader->cache));\n'
                     '        reader->cache_filter = calloc(sets, sizeof(*reader->cache_filter));')
    source = replace(source, 'if (!reader->cache || !reader->miss_lookups',
                     'if (!reader->cache || !reader->cache_filter || !reader->miss_lookups')
    source = replace(source, 'static inline void ple_cache_insert(',
                     'static inline uint64_t ple_cache_filter_bit(int64_t id) {\n'
                     '    uint64_t h = (uint64_t)id * 0x9E3779B97F4A7C15ULL;\n'
                     '    return UINT64_C(1) << ((h >> 48) & 63);\n}\n\n'
                     'static inline void ple_cache_insert(')
    source = replace(source, '    set->ids[w] = id;',
                     '    set->ids[w] = id;\n    uint64_t bits = 0;\n'
                     '    for (unsigned k = 0; k < PLE_CACHE_WAYS; k++) {\n'
                     '        if (set->ids[k] >= 0) bits |= ple_cache_filter_bit(set->ids[k]);\n'
                     '    }\n    reader->cache_filter[set_idx] = bits;')
    source = replace(source, '        if (check_cache) {',
                     '        if (check_cache && (reader->cache_filter[ple_cache_set_index(id, reader->cache_set_mask)] &\n'
                     '                            ple_cache_filter_bit(id))) {')
    return source


def aligned_tags(source):
    source = split_tags(source)
    source = replace(source, '    uint8_t clock;\n    char weights[PLE_CACHE_WAYS][256];\n    char scales[PLE_CACHE_WAYS][8];',
                     '    char weights[PLE_CACHE_WAYS][256];\n    char scales[PLE_CACHE_WAYS][8];\n'
                     '    uint8_t clock;\n    char padding[31];')
    source = replace(source, '        reader->cache = calloc(sets, sizeof(*reader->cache));\n'
                     '        reader->cache_ids = malloc(sets * PLE_CACHE_WAYS * sizeof(*reader->cache_ids));',
                     '        _Static_assert(sizeof(*reader->cache) == 1088, "aligned cache payload stride");\n'
                     '        if (sets > SIZE_MAX / sizeof(*reader->cache)) return EOVERFLOW;\n'
                     '        if (posix_memalign((void **)&reader->cache, 64, sets * sizeof(*reader->cache)))\n'
                     '            reader->cache = NULL;\n'
                     '        else memset(reader->cache, 0, sets * sizeof(*reader->cache));\n'
                     '        if (posix_memalign((void **)&reader->cache_ids, 64, sets * PLE_CACHE_WAYS * sizeof(*reader->cache_ids)))\n'
                     '            reader->cache_ids = NULL;')
    return source


def adaptive_filter(source):
    source = filter_tags(source)
    source = replace(source, '    uint64_t *cache_filter;',
                     '    uint64_t *cache_filter;\n    bool cache_filter_active;')
    source = replace(source, '        reader->cache_filter = calloc(sets, sizeof(*reader->cache_filter));',
                     '        reader->cache_filter = calloc(sets, sizeof(*reader->cache_filter));\n'
                     '        reader->cache_filter_active = true;')
    source = replace(source, '    bool fill_cache = use_cache;',
                     '    uint64_t old_hits = reader->cache_hits, old_misses = reader->cache_misses;\n'
                     '    bool fill_cache = use_cache;')
    source = replace(source, '        if (check_cache && (reader->cache_filter[ple_cache_set_index(id, reader->cache_set_mask)] &\n'
                     '                            ple_cache_filter_bit(id))) {',
                     '        if (check_cache && (!reader->cache_filter_active ||\n'
                     '            (reader->cache_filter[ple_cache_set_index(id, reader->cache_set_mask)] & ple_cache_filter_bit(id)))) {')
    source = replace(source, '    return true;\n}\n\n\n#endif',
                     '    if (check_cache) {\n'
                     '        uint64_t hits = reader->cache_hits - old_hits, misses = reader->cache_misses - old_misses;\n'
                     '        if (hits + misses) reader->cache_filter_active = misses > hits;\n'
                     '    }\n    return true;\n}\n\n\n#endif')
    return source


def soa_tags(source):
    source = split_tags(source)
    source = replace(source, '#define PLE_SPLIT_TAGS 1', '#define PLE_SPLIT_TAGS 1\n#define PLE_SOA_CACHE 1')
    source = replace(source, '    char weights[PLE_CACHE_WAYS][256];\n    char scales[PLE_CACHE_WAYS][8];\n', '')
    source = replace(source, '    int64_t *cache_ids;',
                     '    int64_t *cache_ids;\n    char *cache_weights, *cache_scales;')
    source = replace(source, '        reader->cache_ids = malloc(sets * PLE_CACHE_WAYS * sizeof(*reader->cache_ids));',
                     '        if (sets > SIZE_MAX / (PLE_CACHE_WAYS * 272u)) return EOVERFLOW;\n'
                     '        if (posix_memalign((void **)&reader->cache_ids, 64, sets * PLE_CACHE_WAYS * 272u))\n'
                     '            reader->cache_ids = NULL;\n'
                     '        else {\n'
                     '            reader->cache_weights = (char *)(reader->cache_ids + sets * PLE_CACHE_WAYS);\n'
                     '            reader->cache_scales = reader->cache_weights + sets * PLE_CACHE_WAYS * 256u;\n'
                     '        }')
    for way in ('w', 'hit_way'):
        source = replace(source, f'set->weights[{way}]', f'reader->cache_weights + (set_idx * PLE_CACHE_WAYS + {way}) * 256u')
        source = replace(source, f'set->scales[{way}]', f'reader->cache_scales + (set_idx * PLE_CACHE_WAYS + {way}) * 8u')
    # Lookup no longer needs the clock array; insertion still uses it.
    source = replace(source, '            ple_cache_set_t *set = &reader->cache[set_idx];\n', '')
    return source


def fixture(source):
    source = replace(source, 'static unsigned char byte_at(int64_t id, size_t byte, bool scale) {',
                     'static unsigned char byte_at(int64_t id, size_t byte, bool scale) {\n'
                     '    if (byte < 8) return (unsigned char)((uint64_t)id >> (8 * byte)) ^ (scale ? 0x5a : 0xa5);')
    source = replace(source, '    assert(argc == 6);', '    assert(argc == 8);')
    source = replace(source, '    unsigned rounds = strtoul(argv[5], NULL, 10);',
                     '    unsigned rounds = strtoul(argv[5], NULL, 10);\n'
                     '    unsigned warm_rows = strtoul(argv[6], NULL, 10);\n'
                     '    unsigned repetitions = strtoul(argv[7], NULL, 10);\n'
                     '    assert(warm_rows > 0 && warm_rows < 16000000 && repetitions > 0);')
    source = replace(source, '    if (populated) warm(&p, 1024);',
                     '    uint64_t warm_start = now_ns();\n'
                     '    if (populated) warm(&p, warm_rows);\n'
                     '    uint64_t warm_elapsed = now_ns() - warm_start;')
    source = replace(source, '    uint64_t digest = 1469598103934665603ULL;',
                     '#if defined(PLE_SOA_CACHE)\n'
                     '    size_t extra_bytes = p.cache_sets * PLE_CACHE_WAYS * 272u;\n'
                     '    char *extra = (char *)p.cache_ids;\n'
                     '#elif defined(PLE_SPLIT_TAGS)\n'
                     '    size_t extra_bytes = p.cache_sets * PLE_CACHE_WAYS * sizeof(*p.cache_ids);\n'
                     '    char *extra = (char *)p.cache_ids;\n'
                     '#elif defined(PLE_FILTER_TAGS)\n'
                     '    size_t extra_bytes = p.cache_sets * sizeof(*p.cache_filter);\n'
                     '    char *extra = (char *)p.cache_filter;\n'
                     '#else\n    size_t extra_bytes = 0;\n    char *extra = NULL;\n#endif\n'
                     '    char *extra_before = malloc(extra_bytes ? extra_bytes : 1);\n'
                     '    assert(extra_before);\n'
                     '    if (extra_bytes) memcpy(extra_before, extra, extra_bytes);\n'
                     '    uint64_t digest = 1469598103934665603ULL;\n'
                     '    printf("{\\"warm_rows\\":%u,\\"repetitions\\":%u,\\"extra_cache_bytes\\":%zu,\\"warm_ns\\":%" PRIu64 ",", warm_rows, repetitions, extra_bytes, warm_elapsed);')
    source = replace(source, 'printf("{\\"tokens\\":', 'printf("\\"tokens\\":')
    source = replace(source, '(r >> 8) % 1024 : 1024 + (r >> 8) % 15998976',
                     '(r >> 8) % warm_rows : warm_rows + (r >> 8) % (16000000 - warm_rows)')
    source = replace(source, '        uint64_t start = now_ns();\n        assert(ple_plan(&p, (const char *)ids, count, weights, scales, false));\n        uint64_t elapsed = now_ns() - start;',
                     '        uint64_t start = now_ns();\n'
                     '        for (unsigned repeat = 0; repeat < repetitions; repeat++) {\n'
                     '            p.unique_blocks = p.requested_bytes = p.cache_hits = p.cache_misses = 0;\n'
                     '            assert(ple_plan(&p, (const char *)ids, count, weights, scales, false));\n'
                     '        }\n'
                     '        uint64_t elapsed = (now_ns() - start) / repetitions;')
    source = replace(source, '        assert(memcmp(cache_before, p.cache, p.cache_sets * sizeof(*p.cache)) == 0);',
                     '        assert(memcmp(cache_before, p.cache, p.cache_sets * sizeof(*p.cache)) == 0);\n'
                     '        if (extra_bytes) assert(memcmp(extra_before, extra, extra_bytes) == 0);')
    source = replace(source, '    free(ids); free(weights);', '    free(extra_before); free(ids); free(weights);')
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--cc', default='cc')
    parser.add_argument('--opt', default='-O2')
    parser.add_argument('--real-points', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.base.read_text()
    probe = fixture(Path(__file__).with_name('probe.c').read_text())
    (args.out / 'audit-probe.c').write_text(probe)
    variants = {'control': source, 'probe96': bound(source, 96), 'probe128': bound(source, 128),
                'split64': split_tags(source), 'split96': bound(split_tags(source), 96),
                'filter64': filter_tags(source), 'filter96': bound(filter_tags(source), 96),
                'aligned64': aligned_tags(source), 'aligned96': bound(aligned_tags(source), 96),
                'adaptive64': adaptive_filter(source), 'adaptive96': bound(adaptive_filter(source), 96),
                'soa64': soa_tags(source), 'soa96': bound(soa_tags(source), 96)}
    report = {'scope': 'CPU planner only; allocator destinations, no CUDA mapping or physical I/O',
              'source_base_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'sweep_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'probe_sha256': hashlib.sha256(probe.encode()).hexdigest(), 'platform': platform.platform(),
              'started_unix': time.time(), 'commands': [], 'runs': [], 'variants': {}}

    def command(argv):
        start = time.time()
        frequency = Path('/sys/devices/system/cpu/cpu19/cpufreq/scaling_cur_freq')
        before_khz = frequency.read_text().strip() if frequency.exists() else None
        result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=180)
        report['commands'].append({'argv': list(map(str, argv)), 'exit': result.returncode,
                                   'elapsed_s': time.time() - start, 'stdout': result.stdout, 'stderr': result.stderr,
                                   'cpu19_khz_before': before_khz,
                                   'cpu19_khz_after': frequency.read_text().strip() if frequency.exists() else None})
        result.check_returncode()
        return result.stdout

    try:
        compiler = args.cc
        report['compiler'] = command([compiler, '--version'])
        for name, header in variants.items():
            directory = args.out / name
            directory.mkdir()
            (directory / '_row_plan.h').write_text(header)
            diff = ''.join(difflib.unified_diff(source.splitlines(True), header.splitlines(True),
                                               'a/vllm/models/deepseek_v4_1/engram_rows/_row_plan.h',
                                               'b/vllm/models/deepseek_v4_1/engram_rows/_row_plan.h'))
            (directory / 'change.diff').write_text(diff)
            report['variants'][name] = {'header_sha256': hashlib.sha256(header.encode()).hexdigest()}
            flags = ['-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all'] if args.sanitize else [args.opt]
            command([compiler, *flags, '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-I', directory.resolve(),
                     (args.out / 'audit-probe.c').resolve(), '-o', (directory / 'probe').resolve()])
        # Include threshold boundaries, fully populated/evicting caches and cold starts.
        cases = [(tokens, hot, 1024, True) for tokens in (64, 65, 96, 97, 128) for hot in (0, 75)]
        cases += [(16, 75, 1024, True), (96, 100, 1024, True), (128, 100, 1024, True),
                  (129, 75, 1024, True), (6144, 75, 1024, True), (96, 0, 1024, False)]
        cases += [(tokens, hot, 16384, True) for tokens in (64, 96) for hot in (0, 75, 100)]
        if args.real_points:
            cases = [(tokens, hot, warm_rows, True) for tokens in (6, 8, 48, 96)
                     for hot in (0, 50, 75, 100) for warm_rows in (1024, 16384)]
            cases += [(65, 75, 1024, True), (129, 75, 1024, True), (6144, 75, 1024, True), (96, 0, 1024, False)]
        for tp in (3, 4):
            for tokens, hot, warm_rows, populated in cases:
                results = {}
                names = list(variants)
                if args.reverse:
                    names.reverse()
                for name in names:
                    rounds = 2 if args.sanitize else (31 if tokens <= 128 else 11)
                    # Each timed plan gets distinct IDs. Repeating one batch would
                    # hide the metadata locality cost this experiment examines.
                    repetitions = 1
                    result = json.loads(command([(args.out / name / 'probe').resolve(), tokens, tp, hot,
                                                 int(populated), rounds, warm_rows, repetitions]))
                    results[name] = result
                control = results['control']
                for name, result in results.items():
                    assert result['ids_digest_fnv64'] == control['ids_digest_fnv64']
                    equivalent = results['control' if name.endswith('64') else 'probe96'] if name.startswith(('split', 'filter', 'aligned', 'adaptive', 'soa')) else None
                    if equivalent:
                        for a, b in zip(result['samples'], equivalent['samples']):
                            assert all(a[k] == b[k] for k in ('planned_jobs', 'planned_bytes', 'cache_hits', 'cache_misses'))
                report['runs'].append(results)
    finally:
        report['finished_unix'] = time.time()
        (args.out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'cases': len(report['runs']), 'variants': len(variants), 'receipt': str(args.out / 'receipt.json')}))


if __name__ == '__main__':
    main()
