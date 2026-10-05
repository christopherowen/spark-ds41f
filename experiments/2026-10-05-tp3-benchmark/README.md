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
