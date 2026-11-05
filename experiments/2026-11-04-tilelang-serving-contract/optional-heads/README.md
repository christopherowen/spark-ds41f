# NVFP4 head experiment, excluded from serving

`restore-nvfp4-heads.patch` restores the removed DS4.1 Markov and drafter-owned
NVFP4 head alternatives over the source named in its header. It does not restore
the obsolete Engram setting or the MoE fallbacks. It also removes the two NVFP4
names from the runtime rejection list and returns the corresponding tests to
the earlier supported-option contract.

The patch is deliberately absent from `../vllm/series`. To investigate it,
create another experimental source recipe, append this patch, update its
capabilities and source identifiers, and qualify its own image. The existing
`ds41-serving-contract` doctor rule must not be used to claim a recipe that
restores these options still has the fixed serving contract. No acceptance,
memory, speed or quality qualification is claimed for this experiment.
