FROM docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf
RUN python3 -c "import glob; p=glob.glob('/dev/nvidia*')+glob.glob('/dev/nvhost*')+glob.glob('/dev/dri/*')+glob.glob('/dev/nvmap')+glob.glob('/dev/nvgpu*'); print('GPU device paths:',p); assert not p"
