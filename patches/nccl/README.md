# NCCL patch stack

Base: `NVIDIA/nccl@73cf112295c33aee2b895f329f592f2a9b4b0f97` (release tag
`v2.30.7-1`), the version the runtime base image ships as the
`nvidia-nccl-cu13` wheel. The image rebuilds it for SM121 and replaces the
wheel's `libnccl.so.2`; the image build checks the version and that the
installed library is the one it built.

- `0001-ib-cts-nreqs-acquire-fence.patch` (Stanislav Bardyuk) orders the
  clear-to-send `nreqs` load after the `idx` check in `ncclIbIsend`. The
  receiver's NIC writes the CTS fifo slot by RDMA; on AArch64 the two loads
  can be reordered, so `nreqs` can return the slot's value from a previous
  round and the proxy thread spins forever on a request the receiver never
  writes (a hang with every rank waiting). One acquire fence, a `dmb ishld`
  on AArch64 on the path where the CTS has arrived. Only the IB/RoCE send
  path is touched; RoCEnante carries the small decode collectives and NCCL
  the rest. Upstream: NVIDIA/nccl#2393 (open), fixing NVIDIA/nccl#1983.
  Applying it to the base yields patch head `7522cb27` and tree `47687d2a`.
