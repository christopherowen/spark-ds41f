# Lab image for the #842 crash repro: candidate-37 (sha256:e1ee7b1953b1131320746f87fe0cb11a673b53e9702bda63048ea3922768c15a) with 0676 reverted and the nonfinite census (cddb10983).
FROM spark-ds41f:candidate-37
COPY tests/kernels/moe/test_deepseek_v41_tilelang_router.py /opt/spark-ds41f/candidate/vllm/tests/kernels/moe/test_deepseek_v41_tilelang_router.py
COPY vllm/models/deepseek_v4_1/tilelang/router.py /opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/router.py
COPY vllm/models/deepseek_v4_1/tilelang/sparse_mla.py /opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/sparse_mla.py
COPY vllm/v1/worker/gpu/model_runner.py /opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/model_runner.py
LABEL local.spark-ds41f.lab.purpose="#842 nonfinite census repro (not for serving)" local.spark-ds41f.lab.base="sha256:e1ee7b1953b1131320746f87fe0cb11a673b53e9702bda63048ea3922768c15a" local.spark-ds41f.vllm.patch-head="cddb10983ba7ca08db078a1f9f7f8b0feac6c82a" local.spark-ds41f.vllm.tree="874cbfb25d8b5b4d5e48204f935a74786a4eafeb"
