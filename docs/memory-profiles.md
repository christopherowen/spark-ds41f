# Page-size and context profiles

The serving image and model arithmetic are identical in every shape of the
recipe (`config/cluster.json`). Each shape records its required running
kernel; launch refuses a mismatch. The first two are the three-node (TP3)
shapes; `tp4-ring` is the four-node ring's TP4 1M shape, with 16 sequences, whose
pool holds 11 full 1M contexts at once ([KV capacity](#kv-capacity)).

| Shape | CPU pages | KV allocation per rank | Per-request token limit |
| --- | ---: | ---: | ---: |
| `tp3-triangle-4k` | 4 KiB | 2.2 GiB | 262,144 |
| `tp3-triangle` | 64 KiB, memory-saver enabled | 4.91 GiB | 1,048,576 |
| `tp4-ring` (four nodes) | 64 KiB, memory-saver enabled | 10.5 GiB | 1,048,576 |

**Externally reported, not qualified here:** another site served `tp4-ring` at
this sizing on DGX OS 7.5.0's stock 4 KiB kernel (`6.17.0-1032-nvidia`) without
the memory saver, keeping 18-21 GiB of MemAvailable per node through a 993K-token prefill
([receipt](../experiments/2026-10-17-site-tp4-4k-1017/README.md): one boot, the
site's own battery rather than `bin/spark bench`, no soak). The qualified
`tp4-ring` row above is unchanged.

The limit includes prompt and generated tokens. Eight sequences remain admitted;
the token limit does not promise eight simultaneous full-length contexts. The selected
profile's pool holds five full 1,048,576-token contexts at once, or all eight sequences
at up to 697,600 tokens each (since r6x, whose expert scale codes freed the memory; r6v's
3.5 GiB pool held seven full 524,288-token contexts). vLLM's startup line reports 4,765,898
KV tokens, 4.55 contexts (r6v: 2,845,543, 5.43): it charges every request a prompt's chunks
in flight ([KV capacity](#kv-capacity)). dgx1's 5 GiB startup guard bounds the pool: 5.875
GiB ended below it at graph capture (experiments/2026-10-19-r6x-tp3-1m).

For simultaneous long-context admission checks, use
`bench --admission-tokens 500000 --admission-output-tokens 4096 --admission-force-length`.
The forced output budget keeps early requests active while later requests
prefill. With the usual 256-token replies, an early request can finish before
all four long prompts are admitted, leaving the overlap check inconclusive.
The report records the output budget and whether generation was forced.

The 64 KiB allocation spends 1.3 GiB of the previously measured 1.78–1.86 GiB
memory gain, retaining 0.48–0.56 GiB for additional workspace and variation.
The checkpoint's native limit is 1,048,576 tokens. Since r6x this profile serves the
full range without changing RoPE; a cold 1,000,000-token prefill completes with dgx1 at
5.93 GiB available, above its 3 GiB steady guard.

Select the variant with `bin/spark site shape tp3-triangle-4k --apply` or
`tp3-triangle` (the 64 KiB shape, also the normal GRUB boot default on all
three nodes). Both are entries of the one recipe: `tp3-triangle-4k` extends
`tp3-triangle` with its kernel and sizing only, so nothing else can drift.

## KV capacity

Each rank's pool is a number of 227,840-byte blocks: one block holds 256 tokens of the
compressed caches (the latents and indexer keys of layers 2, 8, 14 and 20), or three
layers' 128-token sliding windows. A request at the 1,048,576-token limit holds 4,142
blocks: 4,096 of compressed caches, three in each of the 15 window groups (the window, the
decode rows of the two steps in flight and the drafter's lookahead) and one of compressor
states. The prompt being filled also holds, in every window group, its chunk in flight and
its next chunk. The pool needs that once, not for every request:

| Shape and budget | Pool blocks | Full-length contexts at once | All sequences at once | vLLM's startup line |
| --- | ---: | ---: | ---: | ---: |
| `tp4-ring`, 16,384 batched tokens | 49,483 | 11 (11.03) | 16 x 718,848 tokens | 6,512,700 tokens (6.21x) |
| `tp4-ring`, 8,192 batched tokens | 49,483 | 11 (11.49) | 16 x 749,568 tokens | 8,580,566 tokens (8.18x) |
| `tp3-triangle`, 4,096 | 23,121 | 5 (5.35) | 8 x 697,600 tokens | 4,765,898 tokens (4.55x) |
| `tp3-triangle-4k` at 262,144 tokens, 4,096 | 10,367 | 8 (8.80) | 8 x 262,144 tokens | 1,348,708 tokens (5.14x) |

vLLM's startup line ("GPU KV cache size", "Maximum concurrency") charges every request the
prompt's two chunks, as if every request prefilled at once, so a larger prefill budget
lowered it (6.21 where 8,192 gave 8.18) although the ring holds 11 at either budget; the
budget's cost is 0.46 of a context's blocks, for the prompt in flight only. vLLM's
`kv-capacity` patch makes the line charge them once (11.02x and 11.48x on the ring). For
any recipe or budget, `scripts/kv_capacity.py [--shape tp4-ring] [--max-num-batched-tokens N]`
prints these figures. The derivation, when two chunks are in flight and the admission
check are in [experiments/2026-10-25-kv-capacity](../experiments/2026-10-25-kv-capacity/README.md).

## Install memory-saver

Reserve the cluster through `scripts/lab.py window open`, then stop all ranks
with `bin/spark --cluster-config <profile> cluster stop --remove --apply --parallel`. From the same clean,
published deployment checkout on each node, run:

```sh
bash experiments/2026-10-02-memory-saver-capacity/install-dkms.sh
```

The script clones the revision pinned in `config/kernel-trial.json`, checks the
existing fleet signing identity, registers only the build inputs, and builds
and installs DKMS 0.4.1 for the pinned 64 KiB kernel. It removes an older
memory-saver registration first (its hook restores the stock UVM and its source
stays in `/usr/src`), and refuses a dirty source checkout or an existing source
directory for the new version rather than overwriting them. It preserves
the system's DKMS signing configuration and stock NVIDIA RM. Inspect the key
paths printed by DKMS; they must be the enrolled fleet key already configured
for fan-control. Kernel prerequisites and both swap files must already be
prepared. Detailed independent setup/removal instructions are in
[memory-saver](https://github.com/christopherowen/dgx-spark-memory-saver).

Boot each node into `7.0.0-1019-nvidia-64k`, using the documented
[one-shot boot procedure](../experiments/2026-10-02-kernel-64k/boot-test/README.md).
Keep serving stopped until every node has 65,536-byte pages, the expected loaded
UVM source identity and `uvm_pack_sysmem_leaf_tables=Y`. Run CUDA validation on
each node before launching the distributed service.

## Doctor and launch checks

`doctor --live` inventories the DKMS registration, selected candidate module,
signature, loaded UVM source identity and packing parameter without loading a
module or initializing CUDA. A missing candidate installation warns even on
the 4 KiB profile; corrective instructions are printed separately. The 64 KiB
profile requires a matching installed and loaded memory-saver module. These
requirements also run before any serving container is launched.

```sh
bin/spark --cluster-config config/cluster.json --shape tp3-triangle doctor --live
bin/spark --cluster-config config/cluster.json --shape tp3-triangle cluster start --apply
```

Before starting, doctor naturally reports absent containers when the service
is stopped. Kernel/module issues must be resolved before launch. The startup
and steady memory guards remain at 5 GiB and 3 GiB respectively.

## Return to the 4 KiB profile

Stop all ranks, select the retained `7.0.0-1019-nvidia` GRUB entry on every
node and reboot. Verify 4,096-byte pages and stock UVM, then start using
the `tp3-triangle-4k` shape. The page-aware memory service selects
the matching swap file and host policy. A 64 KiB DKMS installation can remain
staged: it is excluded from the 4 KiB kernel. Permanent rollback also selects
the 4 KiB production configuration and `config/kernel-trial.json` default through
a published commit, then applies that GRUB default on every node.
