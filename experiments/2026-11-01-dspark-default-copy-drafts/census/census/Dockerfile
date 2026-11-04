# Lab image for the #842 crash repro: candidate-37 (sha256:e1ee7b1953b1131320746f87fe0cb11a673b53e9702bda63048ea3922768c15a) with the nonfinite census (cb46d6c82).
FROM spark-ds41f:candidate-37
COPY tests/kernels/moe/test_deepseek_v41_tilelang_router.py /opt/spark-ds41f/candidate/vllm/tests/kernels/moe/test_deepseek_v41_tilelang_router.py
COPY vllm/models/deepseek_v4_1/tilelang/router.py /opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/router.py
COPY vllm/v1/worker/gpu/model_runner.py /opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/model_runner.py
LABEL local.spark-ds41f.lab.purpose="#842 nonfinite census repro (not for serving)" local.spark-ds41f.lab.base="sha256:e1ee7b1953b1131320746f87fe0cb11a673b53e9702bda63048ea3922768c15a" local.spark-ds41f.vllm.patch-head="cb46d6c82f7e29b901b871d8c8a229998dcaeff4" local.spark-ds41f.vllm.tree="00d1e4ea0fe4248c79eff7fed2435314964c067c"
