# Populate ~48 GiB of realistic clean VFS page cache + interleaved 64K anon churn
# (matches a node that has read model shards / images without drop_caches: MemAvailable
# stays ~121 GiB while MemFree drops so un-defagged base draws 64K..256K fragments).
import glob, mmap, os
MAP_POPULATE = 0x08000
files = sorted(glob.glob("/home/chris/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/*/*.safetensors"))
f = open(files[0], "rb")
fsize = os.path.getsize(files[0])
small_mmaps = []
for i in range(32768):
    m = mmap.mmap(-1, 65536, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_POPULATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    small_mmaps.append(m)
    f.seek((i * 131072) % max(fsize - 65536, 1))
    _ = f.read(65536)
for m in small_mmaps:
    m.close()
f.close()
tot = 0
for fp in files:
    if tot >= (44 << 30):
        break
    with open(fp, "rb") as fh:
        while tot < (44 << 30):
            b = fh.read(16 << 20)
            if not b:
                break
            tot += len(b)
