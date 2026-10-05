# TP3 benchmark: TileLang and B12X

The TP3 acceptance benchmark for both kernel families (8 streams at 512K), on
the dgx1-dgx2-dgx3 triangle, after the [TP4 benchmark](../2026-10-05-tp4-validation/README.md).

**Network.** The CX7 links were re-addressed for the triangle on 2026-10-05:
`cx7-config.py` (fleet repository) found dgx1 port 1 cabled to dgx3 port 0
over LLDP, and dgx1's and dgx3's former links to dgx4 moved from 10.14 and
10.34 to 10.13.{1,2}. All twelve RoCE ports are active with their IPv4 GIDs at
index 3, every path answers, and the backups are in
`/root/spark3-tp3-20261005/` on dgx1 and dgx3. The configs use the triangle
node map `config/nodes-tp3.local.json` (git-ignored, identical on the Mac and
dgx1-dgx3).

| Family | Config | Image | Transport |
| --- | --- | --- | --- |
| B12X | [b12x.json](b12x.json) | `-tp4-1m-v1` | `rocenante-direct` |
| TileLang | [tilelang.json](tilelang.json) | `-tilelang-1m-v3` | `oneshot-direct` (sparknet) |

Both take the promoted TP3 shape from `config/cluster-64k.json` (TP 3, 524,288
tokens, 8 sequences, 4,096-token batches, CUDA graphs to 48 rows, 3.5 GiB of
KV per rank, retention every 512 tokens) and its NCCL settings, and each
family's features from its TP4 candidate: the images, native BF16 drafter
heads, the packed target head and the fabricated-context fix; TileLang adds
its kernel backends. sparknet's one-shot collectives take B12X's TP3 RoCE
sizes (4 MiB all-gather, 2 MiB all-reduce). Each family profiles its own
DSpark cost curves.

## The attention profile's shared block (vLLM 0041)

TileLang's first TP3 start stopped in the DeepSeek V4 attention memory
profile: TileKernels' fused gate refused non-finite logits on five rows, the
drafter's query block. The profile runs a full-budget prefill on a minimal
cache of one block, and every KV cache group used that block at once. The
groups overlay the same memory, so the drafter read the target's records in
its own format: the same fault as 0039, on another dummy layout. TP4 had read
finite garbage there by chance; B12X's router routes non-finite logits
without complaint.

[0041](../2026-10-04-tilelang-1m/vllm/0041-dummy-blocks-per-group.patch)
gives each group its own slice of the pool in every dummy layout (the
fabricated profiling context included), and the attention profile allocates
one block per group. It changes no serving path. Until the next image build,
[tilelang.json](tilelang.json) mounts the two changed files
([overlay](overlay/vllm/v1/worker/gpu/)) over `-tilelang-1m-v3`; the dummy
layout tests pass on that image.

With 0041 the start went on to the drafter's CUDA graph capture and stopped
there the same way. The target marks the rows that pad its batch, and every
row of a dummy batch, as padding, and the MoE routers skip them; the DSpark
drafter set no mask, so its routers took capture's dummy rows, which read
the null block that dummy runs fill with other groups' records.
[0042](../2026-10-04-tilelang-1m/vllm/0042-dspark-drafter-padding-rows.patch)
gives the drafter a persistent padding mask: all padding during capture,
rows past the live query rows on every step, all padding for a profiling
batch. In serving it keeps the drafter's CUDA graph padding rows out of the
routers too, which nothing guaranteed before. The config mounts its file
over the image as well.

## Numerical check

The TP4 benchmark showed TileLang accepting fewer drafts at most points while
its steps were faster. Each family's text is identical across samples, so
the bench's intervals hide the spread between texts. One boot per family runs,
before the benchmark:

- [score_text.py](score_text.py): four fixed texts (Python source and prose,
  2,048 tokens each) scored through prefill; the families compare token for
  token on the same tokens;
- `experiments/2026-09-29-r5k/consistency.py`: each family's decode against
  its own prefill on greedy generations;
- decode acceptance over 13 greedy cases at one stream, so the comparison
  averages over many texts.

## Benchmark

Then, in the same boot: quality, decode on prose and code with reasoning at
1, 2, 4 and 8 streams (three samples), and prefill on real source text at
32K, 256K and 500K (two repeats). B12X compares with the promoted TP3
baseline, TileLang with B12X.

## Results

B12X ran in the window of 2026-10-05 01:03–01:22 UTC. TileLang ran at 01:47–02:08 UTC, after
0041 and 0042; its first three starts stopped as described above. Both
passed quality 5/5 and saw no thermal slowdown.

**Numerical check: no numerical issue.**

| | B12X | TileLang |
| --- | ---: | ---: |
| Same tokens (8,188): mean NLL | 1.5527 | 1.5525 |
| Same tokens: top-1 accuracy | 68.04% | 67.84% |
| Same top token in both families | | 93.89% |
| Decode against prefill: mean logprob gap | 0.0565 | 0.0469 |
| Decode against prefill: argmax differs | 3.32% | 2.59% |
| Accepted drafts per step, mean of 13 texts | 2.384 | 2.401 |
| Single-stream step, mean of 13 texts | 45.66 ms | 43.80 ms |

On the same tokens the two families score the same text equally well, and
TileLang's decode agrees with its own prefill more closely than B12X's.
Across 13 texts acceptance is level (TileLang higher in 7, median ratio
1.01), with a per-text spread of about 5% in either direction. That spread is
why single points of the TP4 benchmark showed lower acceptance: each point is
one text, and TileLang generates the same text in every sample.

**Benchmark** (8 streams at 512K):

| Aggregate tok/s | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| Prose, B12X | 50.4 | 75.9 | 118.8 | 173.3 |
| Prose, TileLang | 50.5 | 78.7 | 122.5 | 181.6 |
| Code, B12X | 60.1 | 90.8 | 136.3 | 192.3 |
| Code, TileLang | 60.9 | 88.3 | 137.6 | 221.6 |

| Prefill, real text (tok/s) | 32K | 256K | 500K |
| --- | ---: | ---: | ---: |
| B12X | 3,778 | 3,632 | 3,432 |
| TileLang | 4,243 | 3,971 | 3,682 |

TileLang's single-stream steps are 5.5% and 5.4% shorter (39.60 and
43.91 ms), eight-stream code is 15.2% faster, and prefill is 7-12% faster.
Code at two streams reads 2.8% lower, inside its interval. Against the
promoted r5o TP3 baseline, the B12X configuration is level within noise.

Memory is tighter than on TP4. During the 500K prefill, the lowest MemAvailable
was 5.9 GiB on dgx1 for B12X and 6.4 GiB for TileLang, with the promoted 3.5
GiB of KV per rank; the startup guard is at 5 GiB.

**Coherence (TileLang).** [coherence.py](coherence.py) answers to six prompts
are coherent and correct: an explanation of hash-map collisions, an
expand-around-centre palindrome function, the arrival time (12:10), a
five-sentence story, a JSON list (FORTRAN 1957, Lisp 1958, C 1972; cut by the
1,200-token budget after long reasoning) and a syllogism ("No"). The repeated
n-grams come from the reasoning drafting the final text. At ~420K tokens,
`long_context.py` found both code words (10% and 90% depth); time to first
token was about 112 s and decode ran at 113-119 tok/s with the context
resident.

**Images.** 0041 and 0042 still need to be built into the TileLang and B12X
images; this run mounted them over `-tilelang-1m-v3`. Neither changes a
serving step except 0042, which keeps the drafter's CUDA graph padding rows
out of the routers.
