# Boot memory at tp3-switched: sixth against seventh build

The seventh build's lowest MemAvailable during a boot on sp5 is 5.0 to 5.1 GiB, against 6.3 to 6.7 GiB for builds five
and six, and the engine's breakdown gives 15.05 GiB outside torch's allocator against 13.78 GiB (fourth build).

One boot of each build, no benchmark, to read from the running engine on sp5, sp6 and sp7: the `memory_breakdown`
line, and each engine process's resident, anonymous, shared and file-backed memory.

- `tp3-switched-b6.json`: the sixth build's image (`spark-ds41f:candidate-55-b6`) with its lock.
- `boot-b6-tp3-switched.json`, `boot-b7-tp3-switched.json`: one boot each.

The lock here is the sixth build's with its own series file (`vllm-series`, the 680 patches before `0808`), so its
patch-set fingerprint is this file's; the patch head and tree are the sixth build's.
