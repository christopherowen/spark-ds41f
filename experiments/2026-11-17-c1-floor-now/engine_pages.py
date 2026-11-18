"""Where a loaded engine's memory sits, from one copy of /proc/kpageflags.

usage: engine_pages.py FLAGS_FILE PAGE_BYTES [IDLE_FLAGS_FILE]

On GB10 the GPU shares system RAM: the driver takes the engine's memory from the kernel's page
allocator as plain kernel pages, so they carry none of the slab, page-table, LRU or free flags.
With an engine loaded that class is the engine (an idle node holds well under 1 GiB of it
outside the boot-time reservations). The driver's allocations are compound pages, so the head
and tail flags give the size of each allocation; runs of adjacent pages give how contiguous the
memory is physically, whatever the allocation size. On a 64 KiB kernel the driver's 64 KiB piece
is a single page with no compound flags: there the engine is the plain kernel pages that were
not plain kernel pages in IDLE_FLAGS_FILE, a map of the same node since its last boot with no
engine loaded."""
import sys
import numpy as np

path, page = sys.argv[1], int(sys.argv[2])
idle_path = sys.argv[3] if len(sys.argv) > 3 else None
flags = np.fromfile(path, dtype=np.uint64)
n = len(flags)
bit = lambda b: (flags >> np.uint64(b)) & np.uint64(1) == 1
ram = ~bit(20) & ~bit(32)
free, slab, pgtable, lru, anon = bit(10) & ram, bit(7), bit(26), bit(5), bit(12)
head, tail = bit(15), bit(16)
other = ram & ~free & ~slab & ~pgtable & ~lru
gib = lambda pages: pages * page / 2**30
print(f"{path.split('/')[-1]}: {gib(ram.sum()):.1f} GiB RAM, {page // 1024} KiB pages: free {gib(free.sum()):.1f}, slab {gib((ram & slab & ~free).sum()):.1f}, "
      f"page tables {gib((ram & pgtable).sum()):.2f}, anonymous {gib((ram & lru & anon).sum()):.1f}, file cache {gib((ram & lru & ~anon & ~slab).sum()):.1f}, "
      f"plain kernel pages {gib(other.sum()):.1f} GiB")

def buckets(sizes_pages, weights, edges_bytes, total):
    out = []
    lo = 0
    for hi in edges_bytes + [None]:
        m = (sizes_pages * page >= lo) & ((sizes_pages * page < hi) if hi else True)
        out.append((lo, hi, weights[m].sum() / total))
        lo = hi
    return out

def label(b):
    return f"{b // 2**30} GiB" if b >= 2**30 else f"{b // 2**20} MiB" if b >= 2**20 else f"{b // 1024} KiB"

def show(title, rows):
    print("  " + title)
    for lo, hi, share in rows:
        name = f"under {label(hi)}" if lo == 0 else f"{label(lo)} and up" if hi is None else f"{label(lo)} to {label(hi)}"
        print(f"    {name:>22}: {100 * share:5.1f}%")

# allocation sizes: a compound allocation is a head page and the tail pages that follow it
o = other.astype(np.int8)
idx = np.flatnonzero(other & head)
tails = other & tail
# length of the tail run after each head
edges = np.flatnonzero(np.diff(np.concatenate(([0], tails.astype(np.int8), [0]))))
starts, ends = edges[::2], edges[1::2]            # tail runs [start, end)
run_of = dict(zip(starts.tolist(), (ends - starts).tolist()))
sizes = np.array([1 + run_of.get(i + 1, 0) for i in idx.tolist()], dtype=np.int64)
single = int((other & ~head & ~tail).sum())
total = other.sum()
print(f"  plain kernel pages: {gib(sizes.sum()):.1f} GiB in {len(sizes)} compound allocations, {gib(single):.2f} GiB in single pages")
vals, cnt = np.unique(sizes, return_counts=True)
top = np.argsort(-(vals * cnt))[:6]
print("  allocation sizes holding the most memory: " + "; ".join(f"{label(int(vals[i]) * page)} x {int(cnt[i])} = {gib(int(vals[i] * cnt[i])):.1f} GiB" for i in top))

# physical contiguity of the driver's memory: runs of adjacent pages of its allocation size
unit = int(vals[np.argmax(vals * cnt)])
drv = np.zeros(n, dtype=bool)
hs = idx[sizes == unit]
for k in range(unit):
    drv[hs + k] = True
if idle_path:
    idle = np.fromfile(idle_path, dtype=np.uint64)[:n]
    ib = lambda b: (idle >> np.uint64(b)) & np.uint64(1) == 1
    idle_other = ~ib(20) & ~ib(32) & ~ib(10) & ~ib(7) & ~ib(26) & ~ib(5)
    drv = other & ~idle_other
    print(f"  taking the plain kernel pages that were not such on the idle node as the engine: {gib(drv.sum()):.1f} GiB "
          f"({gib((other & idle_other).sum()):.2f} GiB were already plain kernel pages)")
elif unit * cnt[np.argmax(vals * cnt)] < 0.5 * total:   # no engine loaded: describe the whole class
    drv = other.copy()
else:
    print(f"  taking the {label(unit * page)} allocations as the engine: {gib(drv.sum()):.1f} GiB")
other, total = drv, drv.sum()
o = other.astype(np.int8)
e = np.flatnonzero(np.diff(np.concatenate(([0], o, [0]))))
rs, re_ = e[::2], e[1::2]
rl = re_ - rs
show("share of that memory by length of the contiguous physical run it sits in",
     buckets(rl, rl.astype(np.float64), [64 * 2**10, 256 * 2**10, 2 * 2**20, 32 * 2**20, 512 * 2**20], float(total)))
for blk in (2 * 2**20, 32 * 2**20, 512 * 2**20):
    per = blk // page
    m = n // per
    whole = other[: m * per].reshape(m, per).all(1).sum()
    print(f"  share inside aligned {label(blk)} blocks that are wholly this class: {100 * whole * per / total:.1f}%")
