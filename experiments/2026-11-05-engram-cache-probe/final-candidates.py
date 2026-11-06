import datetime as dt
from pathlib import Path
import subprocess

root = Path('/home/chris/projects/engram-cache-probe')
source = root / 'vllm'
base = '1f33102547e4da5c64712390604695d1abd577c4'
header_path = Path('vllm/models/deepseek_v4_1/engram_rows/_row_plan.h')
out = root / 'final-candidates'
out.mkdir(exist_ok=True)
for label, variant in (('aligned', 'aligned64'), ('filter', 'filter64')):
    tree = root / ('vllm-' + label)
    subprocess.run(['git', '-C', str(source), 'worktree', 'add', '-b', 'feat/engram-cache-' + label, str(tree), base], check=True)
    header = (root / 'runs/layout-a' / variant / '_row_plan.h').read_text()
    header = header.replace('#define PLE_SPLIT_TAGS 1\n', '').replace('#define PLE_FILTER_TAGS 1\n', '')
    if label == 'aligned':
        header = header.replace('    char padding[31];',
                                '    /* Keep every payload set on a cache-line boundary. */\n'
                                '    char padding[64u - ((PLE_CACHE_WAYS * 264u + 1u) % 64u)];')
        header = header.replace('_Static_assert(sizeof(*reader->cache) == 1088, "aligned cache payload stride");',
                                '_Static_assert(sizeof(*reader->cache) % 64u == 0, "aligned cache payload stride");')
    else:
        header = header.replace('static inline uint64_t ple_cache_filter_bit(int64_t id) {',
                                '/* Exact row IDs remain authoritative. This four-bit-at-most summary\n'
                                ' * rejects most misses before touching the much larger payload array.\n'
                                ' * A collision only falls through to the ordinary exact comparison. */\n'
                                'static inline uint64_t ple_cache_filter_bit(int64_t id) {')
    (tree / header_path).write_text(header)
    subprocess.run(['git', '-C', str(tree), 'diff', '--check'], check=True)
    subprocess.run(['git', '-C', str(tree), 'add', str(header_path)], check=True)
    rationale = ('Keep the four exact row IDs in a compact aligned array and align the\n'
                 'immutable weight/scales payload sets to 64 bytes. The default cache\n'
                 'allocations increase from 17.125 MiB to 17.5 MiB per reader.\n') if label == 'aligned' else (
                 'Add a compact 64-bit per-set miss filter, rebuilt from the current\n'
                 'four exact IDs after insertion. A filter collision always falls\n'
                 'through to exact ID comparison. The extra allocation is 128 KiB\n'
                 'per default reader.\n')
    message = (f'Reduce Engram cache lookup cost with {label} metadata\n\n'
               'Upstream base: local-inference-lab/vllm\n'
               '19f2c20ed4d64ae22bd009ceacadd85e2a81e354, after candidate-38\n'
               '1f33102547e4da5c64712390604695d1abd577c4.\n\n'
               'Rationale: probing IDs interleaved with cached payloads costs CPU\n'
               'time on populated-cache misses.\n' + rationale + '\n'
               'Quality: preserve row bytes, exact ID matching, capacity, four-way\n'
               'replacement, admission/probe limits and the reader schedule. No\n'
               'model arithmetic, quantization or batch-dependent numerical change.\n\n'
               'Tests: thirteen-layout native CPU screen on GB10, forward and reverse\n'
               'process order, including TP3 6/48-row and TP4 96-row points, cold\n'
               'lookups and eviction-heavy cache states. The final source is queued\n'
               'for exact-source native reader, sanitizer and GPU validation; serving\n'
               'boot/repeat qualification is pending. No serving gain is claimed.\n\n'
               'Upstream status: experimental local patch; not submitted upstream.\n')
    (out / (label + '-message.txt')).write_text(message)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    import os
    env = dict(os.environ, GIT_AUTHOR_DATE=stamp, GIT_COMMITTER_DATE=stamp)
    subprocess.run(['git', '-C', str(tree), '-c', 'user.name=Christopher Owen',
                    '-c', 'user.email=3221756+christopherowen@users.noreply.github.com',
                    '-c', 'commit.gpgsign=false', 'commit', '-F', str(out / (label + '-message.txt'))], check=True, env=env)
    patch = subprocess.check_output(['git', '-C', str(tree), 'format-patch', '-1', '--stdout'])
    (out / (label + '.patch')).write_bytes(patch)
    identity = subprocess.check_output(['git', '-C', str(tree), 'show', '-s', '--format=commit %H%ntree %T%nparent %P'], text=True)
    (out / (label + '-identity.txt')).write_text(identity)
    print(label, identity, flush=True)
