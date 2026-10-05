# Page-size and context profiles

The serving image and model arithmetic are identical in both profiles. Each
configuration records its required running kernel; launch refuses a mismatch.

| Configuration | CPU pages | KV allocation per rank | Per-request token limit |
| --- | ---: | ---: | ---: |
| `config/cluster-4k.json` | 4 KiB | 2.2 GiB | 262,144 |
| `config/cluster-64k.json` | 64 KiB, memory-saver enabled | 3.5 GiB | 524,288 |

The limit includes prompt and generated tokens. Eight sequences remain admitted;
the token limit does not promise eight simultaneous full-length contexts. KV
capacity and admission are measured at startup and under load. The selected
profile reports 2,845,543 KV tokens, or 5.43 full 524,288-token contexts.

For simultaneous long-context admission checks, use
`bench --admission-tokens 500000 --admission-output-tokens 4096 --admission-force-length`.
The forced output budget keeps early requests active while later requests
prefill. With the usual 256-token replies, an early request can finish before
all four long prompts are admitted, leaving the overlap check inconclusive.
The report records the output budget and whether generation was forced.

The 64 KiB allocation spends 1.3 GiB of the previously measured 1.78–1.86 GiB
memory gain, retaining 0.48–0.56 GiB for additional workspace and variation.
The checkpoint's native limit is 1,048,576 tokens. This profile increases the
configured limit to 524,288 and validates that range without changing RoPE.

Use `--cluster-config config/cluster-4k.json` or
`--cluster-config config/cluster-64k.json` before the subcommand. `config/cluster.json`
selects the validated 64 KiB profile, also the normal GRUB boot default on
all three nodes. Changes to the selected profile must update
both its named file and `cluster.json` together.

## Install memory-saver

Reserve the cluster through `scripts/lab.py window open`, then stop all ranks
with `bin/spark cluster stop --remove --apply --parallel`. From the same clean,
published deployment checkout on each node, run:

```sh
bash experiments/2026-10-02-memory-saver-capacity/install-dkms.sh
```

The script clones the revision pinned in `config/kernel-trial.json`, checks the
existing fleet signing identity, registers only the build inputs, and builds
and installs DKMS 0.2.0 for the pinned 64 KiB kernel. It refuses a dirty source
checkout or existing registration rather than overwriting them. It preserves
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
bin/spark --cluster-config config/cluster-64k.json doctor --live
bin/spark --cluster-config config/cluster-64k.json cluster start --apply
```

Before starting, doctor naturally reports absent containers when the service
is stopped. Kernel/module issues must be resolved before launch. The startup
and steady memory guards remain at 5 GiB and 3 GiB respectively.

## Return to the 4 KiB profile

Stop all ranks, select the retained `7.0.0-1019-nvidia` GRUB entry on every
node and reboot. Verify 4,096-byte pages and stock UVM, then start using
`--cluster-config config/cluster-4k.json`. The page-aware memory service selects
the matching swap file and host policy. A 64 KiB DKMS installation can remain
staged: it is excluded from the 4 KiB kernel. Permanent rollback also selects
the 4 KiB production configuration and `config/kernel-trial.json` default through
a published commit, then applies that GRUB default on every node.
