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

`nf-s-r4c`, `nf-s-r4c-b` (lean screen; the first boot also ran cold prefill):
the image build's checks passed (installed NCCL reports 2.30.7 through
`ncclGetVersion` and matches the built library's checksum); LRU 5/5 on both
boots; cold prefill 3,878-3,894 tok/s at 2K and 4,103-4,207 tok/s at 32K,
where the all-reduces run on NCCL; dgx1's lowest MemAvailable 6.45-6.56
GiB. Single-stream steps 51.37/53.53/51.73/53.72 ms (prose, code,
prose-nothink, code-nothink) against a same-morning lean r4a control
50.68/53.16/51.77/53.60 ms: within noise except prose (+0.7 ms), and even
at eight streams. The patched NCCL costs nothing measurable. (The first
build failed its own check: `torch.cuda.nccl.version()` reports the NCCL
torch was compiled against, 2.29.7; the check now asks the library.)
