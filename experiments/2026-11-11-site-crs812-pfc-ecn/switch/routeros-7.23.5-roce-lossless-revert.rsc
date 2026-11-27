# Revert routeros-7.23.5-roce-lossless.rsc, keeping RouterOS 7.23.5. Stop serving; use Safe Mode.
# Leaves the fabric ports at MTU 9000.
/interface ethernet switch qos port set [find name="<FABRIC_PORT_1>"] map=default trust-l3=ignore pfc=disabled
/interface ethernet switch qos port set [find name="<FABRIC_PORT_2>"] map=default trust-l3=ignore pfc=disabled
/interface ethernet switch qos port set [find name="<FABRIC_PORT_3>"] map=default trust-l3=ignore pfc=disabled
/interface ethernet switch qos port set [find name="<FABRIC_PORT_4>"] map=default trust-l3=ignore pfc=disabled
/interface ethernet switch qos port set [find name="<FABRIC_PORT_1>"] !egress-rate-queue3
/interface ethernet switch qos port set [find name="<FABRIC_PORT_2>"] !egress-rate-queue3
/interface ethernet switch qos port set [find name="<FABRIC_PORT_3>"] !egress-rate-queue3
/interface ethernet switch qos port set [find name="<FABRIC_PORT_4>"] !egress-rate-queue3
/interface ethernet switch qos map ip remove [find map=roce-lossless]
/interface ethernet switch qos map remove [find name=roce-lossless]
/interface ethernet switch qos priority-flow-control remove [find name=roce-lossless]
/interface ethernet switch qos profile remove [find name=roce-lossless]
/interface ethernet switch qos profile remove [find name=roce-cnp]
/interface ethernet switch qos tx-manager queue set [find tx-manager=default traffic-class=3] ecn=no
/interface ethernet switch qos settings set lossless-traffic-class=auto lossless-buffers=auto
/interface ethernet switch qos port print detail without-paging where name~"<FABRIC_PORT_PATTERN>"
