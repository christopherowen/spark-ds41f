"""What sits inside a node's large memory blocks, from /proc/kpageflags and /proc/kpagecount.

usage: blocks.py NAME PAGE_BYTES BLOCK_MIB

A page is free when the kernel flags it as part of a free buddy block; otherwise it is slab,
a page table, anonymous memory, file cache (on an LRU list, not anonymous), or another
kernel allocation (none of those flags: drivers, vmalloc, stacks, per-CPU lists). Blocks are BLOCK_MIB of physical memory."""
import sys
import numpy as np

name, page, block_mib = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
flags = np.fromfile(f"{name}.flags", dtype=np.uint64)
count = np.fromfile(f"{name}.count", dtype=np.uint64)
n = min(len(flags), len(count))
flags, count = flags[:n], count[:n]
bit = lambda b: (flags >> np.uint64(b)) & np.uint64(1) == 1
nopage, reserved = bit(20), bit(32)
ram = ~nopage & ~reserved
slab, pgtable, lru, anon, buddy = bit(7), bit(26), bit(5), bit(12), bit(10)
free = ram & buddy  # the kernel flags every page of a free buddy block
cls = np.full(n, 5, dtype=np.int8)  # other kernel: in use with none of the flags below
cls[~ram] = -1
cls[free] = 0
cls[ram & ~free & slab] = 1
cls[ram & ~free & pgtable] = 2
cls[ram & ~free & ~slab & ~pgtable & lru & anon] = 3
cls[ram & ~free & ~slab & ~pgtable & lru & ~anon] = 4
names = ["free", "slab", "page tables", "anonymous", "file cache", "other kernel"]
gib = lambda pages: pages * page / 2**30
print(f"{name}: {gib(ram.sum()):.1f} GiB of RAM in {page // 1024} KiB pages; " + ", ".join(f"{names[c]} {gib((cls == c).sum()):.2f} GiB" for c in range(6)))
per = block_mib * 2**20 // page
first = int(np.flatnonzero(ram)[0]) // per * per
m = (n - first) // per
blk = cls[first : first + m * per].reshape(m, per)
is_ram = (blk >= 0).sum(1)
full = is_ram == per
counts = np.stack([(blk == c).sum(1) for c in range(6)], axis=1)[full]
B = int(full.sum())
used = per - counts[:, 0]
unmov = counts[:, 1] + counts[:, 2] + counts[:, 5]   # slab, page tables, other kernel: not movable
mov = counts[:, 3] + counts[:, 4]
print(f"  {B} whole blocks of {block_mib} MiB ({B * block_mib / 1024:.0f} GiB): entirely free {int((used == 0).sum())}; "
      f"holding only movable pages {int(((unmov == 0) & (used > 0)).sum())}; holding unmovable pages {int((unmov > 0).sum())}")
spoiled = unmov > 0
if spoiled.any():
    u = unmov[spoiled]
    q = np.percentile(u, [10, 50, 90])
    print(f"  in the blocks with unmovable pages: {gib(u.sum()):.2f} GiB unmovable over {spoiled.sum() * block_mib / 1024:.0f} GiB of blocks "
          f"({100 * u.sum() / (spoiled.sum() * per):.1f}% of them); unmovable pages a block p10/median/p90: "
          + ", ".join(f"{int(v)} ({v * page / 2**20:.1f} MiB)" for v in q))
    few = [int((u <= k).sum()) for k in (1, 4, 16, 64)]
    print(f"  blocks held by at most 1 / 4 / 16 / 64 unmovable pages: {few}")
    tot = counts[spoiled][:, [1, 2, 5]].sum(0)
    print("  the unmovable pages there: " + ", ".join(f"{nm} {gib(t):.2f} GiB" for nm, t in zip(("slab", "page tables", "other kernel"), tot)))
    onlyone = [(int(((counts[spoiled][:, 1] > 0) & (counts[spoiled][:, 2] == 0) & (counts[spoiled][:, 5] == 0)).sum())),
               int(((counts[spoiled][:, 5] > 0) & (counts[spoiled][:, 1] == 0) & (counts[spoiled][:, 2] == 0)).sum())]
    print(f"  blocks whose unmovable pages are slab only: {onlyone[0]}; other kernel only: {onlyone[1]}")
    freeshare = counts[spoiled][:, 0].sum() / (spoiled.sum() * per)
    print(f"  free inside those blocks: {100 * freeshare:.0f}% ({gib(counts[spoiled][:, 0].sum()):.1f} GiB that cannot form a whole block)")
