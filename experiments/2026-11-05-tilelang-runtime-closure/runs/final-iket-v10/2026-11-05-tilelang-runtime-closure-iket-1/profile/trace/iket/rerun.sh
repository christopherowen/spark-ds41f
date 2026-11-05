export CUTE_DSL_COMPILER_OPT=iket
export SMODEL_DEBUG_BREAK=1
export CUDA_INJECTION64_PATH=/out/2026-11-05-tilelang-runtime-closure-iket-1/profile/injection/profiler/libsmodel_injection.so
export SMODEL_INJECTION_CONFIG=/out/2026-11-05-tilelang-runtime-closure-iket-1/profile/trace/iket/iket.config.json
# Working dir 
cd /vllm-workspace
# Execute cmd
/tmp/.venv/bin/python /r/experiments/2026-11-05-tilelang-runtime-closure/iket_probe.py --out /out/2026-11-05-tilelang-runtime-closure-iket-1/probe
