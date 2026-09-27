# NCCL with the AArch64 IB send-path fence (r4c candidate)

Base: the promoted configuration on candidate image
`vllm-ds41f-kkref:01f1b874c774-r4c`: the r4b sources (vLLM `01f1b874` with
patches 0001-0009, B12X `0f846212` with the switchless patch) plus NCCL
2.30.7 rebuilt with `patches/nccl` (tree `47687d2a`) in place of the base
image's wheel.

## Change

NVIDIA/nccl#2393: an acquire fence between the clear-to-send `idx` check and
the `nreqs` load in `ncclIbIsend`. Without it an AArch64 proxy thread can
read a stale `nreqs` and spin forever, hanging every rank. It is a
reliability fix, not a speed change; the arm also drops
`--default-chat-template-kwargs {"thinking":true}` like the other candidate
arms and runs the r4b defaults (three drafts, patches 0005-0009 off).

## Gates

The image build checks that `torch.cuda.nccl.version()` is (2, 30, 7) and
that the installed `libnccl.so.2` is the built one. One boot: LRU 5/5, the
lean decode screen, and cold prefill at 2K and 32K (32K prefill
all-reduces exceed RoCEnante's 2 MB limit, so they run on NCCL). Throughput
must match the r4a base within noise.

## Results

Pending.
