#!/bin/bash
# usage: sass_gate.sh [--expect-fenced]   (on dgx1, deployment checkout, a cluster running)
# Disassembles the running image's compile-cache builds of the kernels B12X 0005
# fences (BF16 prefill projection, mHC TF32 and BF16 TMA projections, contiguous
# attention), selected by the running B12X package fingerprint, and runs the
# audit's scoreboard checker (experiments/2026-09-30-proxy-fence-audit/
# sass_pending.py). The cache is root-owned, so the objects are gathered in the
# container; the image lacks nvdisasm, so cuobjdump runs on the host. With
# --expect-fenced, exits non-zero if any mbarrier arrive in those kernels has
# shared loads pending or if no BF16 prefill or mHC TF32 build is found.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
A=experiments/2026-09-30-proxy-fence-audit
C=dsv41-karmic-kraken
W=/tmp/sass_gate
docker exec $C sh -c 'rm -rf /tmp/sass && mkdir -p /tmp/sass'
docker cp $A/extract_fatbins.py $C:/tmp/sass/
docker exec $C sh -c '
set -e
FP=$(cd /opt/spark3/candidate/b12x && python3 -c "from b12x._lib.compiler import b12x_package_fingerprint as f; print(f())")
echo "b12x fingerprint $FP"
cd /tmp/sass
n=0
for j in $(grep -l -F "$FP" /cache/kkref/jit/b12x/compile/*/*.json); do
  cls=$(grep -o -E "\"(Bf16PrefillKernel|MHCPrefillTf32ProjectTmaKernel|MHCPrefillBf16ProjectTmaKernel|ContiguousAttentionForwardKernel)\"" "$j" | head -1 | tr -d "\"")
  [ -n "$cls" ] || continue
  n=$((n + 1))
  cp "${j%.json}.o" "$cls-$n.o"
done
echo "builds $n"
python3 extract_fatbins.py ./*.o
' || { echo "sass gate: extraction failed"; exit 1; }
rm -rf $W && docker cp $C:/tmp/sass $W
for f in $W/*.fatbin; do /usr/local/cuda-13.0/bin/cuobjdump -sass "$f" > "${f%.fatbin}.sass"; done
python3 $A/sass_pending.py $W/*.sass | tee $W/report.txt
if [ "${1:-}" = --expect-fenced ]; then
  bad=$(grep -c "^  PENDING" $W/report.txt)
  have_bf16=$(grep -c "^== .*Bf16PrefillKernel" $W/report.txt)
  have_mhc=$(grep -c "^== .*MHCPrefillTf32ProjectTmaKernel" $W/report.txt)
  echo "sass gate: pending arrives $bad, BF16 prefill builds $have_bf16, mHC TF32 builds $have_mhc"
  [ "$bad" = 0 ] && [ "$have_bf16" -gt 0 ] && [ "$have_mhc" -gt 0 ] || exit 1
fi
