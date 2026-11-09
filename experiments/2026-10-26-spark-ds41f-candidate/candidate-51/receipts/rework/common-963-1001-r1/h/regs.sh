#!/bin/bash
# usage (in the image, no GPU): bash /h/regs.sh /c/cache  NAME...
# For every cached TileLang kernel whose device source defines one of NAME: its threads
# (__launch_bounds__), dynamic shared memory (host launch), cp.async wait depths, and
# its registers and spills from cuobjdump of the cached executable.
C=$1; shift
for name in "$@"; do
  for cu in $(grep -l "void __launch_bounds__([0-9]*, *[0-9]*) ${name}_kernel\|${name}_kernel(" $C/*/linux-aarch64/kernels/*/device_kernel.cu 2>/dev/null); do
    d=$(dirname $cu)
    lb=$(grep -o "__launch_bounds__([0-9]*" $cu | head -1 | tr -dc 0-9)
    smem=$(grep -o "MaxDynamicSharedMemorySize, *[0-9]*" $d/host_kernel.cu | head -1 | tr -dc 0-9)
    waits=$(grep -o "cp_async_wait<[0-9]*>" $cu | sort -u | tr -dc '0-9 \n' | tr '\n' ' ')
    reg=$(/usr/local/cuda/bin/cuobjdump --dump-resource-usage $d/executable.so 2>/dev/null | grep -A1 "Function ${name}_kernel" | grep -o "REG:[0-9]*\|LOCAL:[0-9]*" | tr '\n' ' ')
    echo "$name threads=$lb dyn_smem=$smem waits=[$waits] $reg dir=$(basename $d | cut -c1-12)"
  done
done
