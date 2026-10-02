# Experimental ABI-compatible CUDA library/JIT-assembler arm; native extensions stay r5o.
FROM vllm-ds41f-kkref:04c30fa98e79-r5o
COPY *.whl /tmp/cuda133-wheels/
COPY cuda-requirements.txt /tmp/cuda133-requirements.txt
RUN python3 -m pip install --no-deps --no-index --find-links=/tmp/cuda133-wheels --require-hashes -r /tmp/cuda133-requirements.txt && rm -r /tmp/cuda133-wheels
COPY ptxas /opt/spark3/cuda133/bin/ptxas
ENV TRITON_PTXAS_BLACKWELL_PATH=/opt/spark3/cuda133/bin/ptxas
LABEL local.spark3.cuda.experiment="13.3-libraries-triton-assembler"
