# Decision

Status: **retain for more evidence** (external site receipt)

On a MikroTik CRS812, PFC and ECN only took effect after moving to RouterOS 7.23.5 with an explicit
queue-3 rate on each fabric port. With that in place, spark-ds41f's NIC retransmissions (`roce_adp_retrans`)
went from 1,425 to zero, and the switch counted no queue drops afterwards (the before run did not record
switch counters). spark-ds41f's cold prefill rose 14% at 4K and 3% at 32K and was unchanged at 128K;
decode did not change.

For this repository it is a worked switch example for [switched-fabric.md](../../docs/switched-fabric.md),
not a recipe change. It is one boot per side with the site's own client, C1 only, on a site branch over
`c460d1b4`, and firmware and QoS changed together. Raw receipts: see `runs/README.md`.
