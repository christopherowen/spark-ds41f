# Memory-saver capacity deployment

Owner-authorized deployment of DKMS memory-saver on all three nodes, with
explicit 4 KiB and 64 KiB serving profiles. Source identity and intended deltas
are in `base.json`. The r5o container image, weights and model arithmetic remain
unchanged. This tests the combined host-memory and capacity change; it does not
attribute speed differences to a single setting.

The larger profile allocates 3.5 GiB KV per rank and accepts up to 524,288 tokens
per request. Acceptance requires expected DKMS and loaded module identities on
all nodes, correct swap/memory policy, standalone CUDA checks, live doctor,
serving quality, decode/prefill/prefix/admission screens, and retrieval beyond
the previous 262,144-token limit. Keep all runs, including failures, under
`results/private/memory-saver-capacity/` before recording the final decision here.

Host installation follows `install-dkms.sh`; operating instructions are in
[`docs/memory-profiles.md`](../../docs/memory-profiles.md). The initial published
preparation leaves the 4 KiB production configuration and boot default intact.
Switch the default only after the 64 KiB profile passes validation.
