#!/bin/bash
# usage: test_overlay.sh   (on dgx1; runs the SP tests on dgx3 in the r5f image)
set -eu
O=spark3-overlay/r5f-sp
V=/opt/spark3/candidate/vllm
ssh dgx3 "M=''; for f in \$(cd ~/$O && find vllm tests -name '*.py'); do M=\"\$M -v \$HOME/$O/\$f:$V/\$f:ro\"; done; \
  timeout 900 docker run --rm --gpus=all \$M --entrypoint bash vllm-ds41f-kkref:04c30fa98e79-r5f -c \
  'pip install -q pytest >/dev/null 2>&1; cd $V && python3 -m pytest -q -p no:cacheprovider --noconftest tests/models/test_deepseek_v4_1_sp_prefill.py tests/models/test_deepseek_v4_1_ced_model.py 2>&1 | tail -15; python3 -c \"import vllm.models.deepseek_v4_1.nvidia.model\" && echo import-ok'"
