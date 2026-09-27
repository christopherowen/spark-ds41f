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

## Workload and gates

`sequence.sh`: current, always, current, always (ABAB). Each runs the LRU
gate, decode at one and eight streams (eight-stream runs mix new prompts
into decode batches), and cold prefill of real text at about 64, 128, 256,
512, 1024, 1536, 2048 and 4096 tokens, four repeats each.

## Results

Pending.
