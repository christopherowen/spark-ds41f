# Four-node forwarding queue capacity

Base: `bc67e13`, candidate transport image
`sha256:488fed96fecec12e60a32a75f387e057bbf56da384098ca6efb6757869734c2d`.

Hypothesis: the installed hairpin queue capacity limits larger forwarded
collectives. Change only `hairpin_queue_size` from 1024 to 8192 on the 16 data
functions, apply it with driver reinitialization while all four nodes are idle,
and compare the same image, graph workload and RDMA/port counter deltas.
Queue count remains four. Model kernels, transport source, payload sizes,
physical topology and every other NIC setting remain unchanged.

Run baseline, candidate, then restore-and-repeat baseline to separate queue size
from a driver-reload effect. Verify effective reload counters, GID/address/MTU,
hardware forwarding and exact collective results. Record every attempt. A
configured driverinit value alone does not establish the active queue size.

The helper requires an owned cluster window, idle GPUs/containers/RDMA contexts,
no TC filters and an exact before-inventory match. It changes only mapped data
functions, never management networking. The coordinator checks all four hosts
before invoking it and verifies all four after the coordinated transition.
An interrupted transition needs inspection and restoration before GPU work.
Restore the entry queue value before releasing the window. Full model-serving
promotion remains separate from this network experiment.
