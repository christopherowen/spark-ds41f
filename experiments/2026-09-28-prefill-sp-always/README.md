# Sequence-parallel prefill on every prompt chunk

Base: the promoted configuration (`2026-09-27-karmic-kraken-r5g`).

## Question

r5g runs prefill sequence-parallel only for forwards of at least 2048
tokens (`SPARK3_DS41_PREFILL_SP_MIN_ROWS`). That figure came from another
stack, not from a measurement here. The only structural limit is decode:
decode forwards (at most 48 tokens) replay captured CUDA graphs and never
use SP. Does SP pay on every other forward, down to small prompt chunks and
prompt chunks mixed into decode batches? If it does, the setting goes away.

## Arms

| Arm | Change from the promoted configuration |
|---|---|
| `current` | none (SP from 2048 tokens) |
| `always` | `SPARK3_DS41_PREFILL_SP_MIN_ROWS=1`, which the code raises to 49: every forward above the decode graph sizes |
| `always2` | vLLM patch 0018 mounted over the r5g image: the setting is gone, SP runs above the decode sizes, and small reduce-scatters use the RoCE all-reduce |

## Workload and gates

`sequence.sh`: current, always, current, always (ABAB). Each runs the LRU
gate, decode at one and eight streams (eight-stream runs mix new prompts
into decode batches), and cold prefill of real text at about 64, 128, 256,
512, 1024, 1536, 2048 and 4096 tokens, four repeats each.

## Results

First round (2026-09-27 23:10-23:36 UTC, ABAB), prefill TTFT in ms:

| Tokens | 73 | 154 | 251 | 478 | 1060 | 1431 | 1776 | 3772 |
|---|---|---|---|---|---|---|---|---|
| current | 227.0 | 249.0 | 287.5 | 345.0 | 466.5 | 552.5 | 639.5 | 976.0 |
| always | 236.0 | 250.0 | 289.5 | 341.0 | 450.5 | 519.0 | 609.0 | 980.0 |
| change | +4.0% | +0.4% | +0.7% | -1.2% | -3.4% | -6.1% | -4.8% | +0.4% |

SP gains from about 500 tokens: the 2048-token cut was leaving 3-6% on
1-2K prompts. Below that it lost up to 9 ms: each encoder layer's two
reduce-scatters went through NCCL at 100-200 us per call on small
messages, against about 16 us for the RoCE all-reduce. Eight-stream decode,
which mixes short prompts into decode batches, read 1-3% lower. Patch 0018
routes those small reduce-scatters through the RoCE all-reduce (`always2`).

Second round: pending.
