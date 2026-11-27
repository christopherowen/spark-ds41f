# Lossless RoCE v2 class for a switched spark-ds41f fabric: MikroTik CRS812, RouterOS 7.23.5.
#
# Replace <FABRIC_PORT_1> .. <FABRIC_PORT_4> with the switch interfaces cabled to the Sparks
# (one 200G port per node). Add or remove lines for other node counts.
# Upgrade BOTH the routeros and switch-marvell packages to 7.23.5 first: before 7.23 the CRS8xx
# accepts these settings without implementing PFC or ECN.
# Stop serving and use Safe Mode. Only switch QoS and fabric-port MTU change here; no addressing,
# bridge, routing or management settings.
:if (!([/system resource get version]~"^7[.]23[.]5")) do={:error "Written for RouterOS 7.23.5"}

# Jumbo frames on the fabric ports (hosts run MTU 9000).
/interface ethernet set [find name="<FABRIC_PORT_1>"] l2mtu=9216 mtu=9000
/interface ethernet set [find name="<FABRIC_PORT_2>"] l2mtu=9216 mtu=9000
/interface ethernet set [find name="<FABRIC_PORT_3>"] l2mtu=9216 mtu=9000
/interface ethernet set [find name="<FABRIC_PORT_4>"] l2mtu=9216 mtu=9000

# Traffic class 3 is the lossless class; let the switch size its buffers.
/interface ethernet switch qos settings set shared-buffers=auto multicast-buffers=auto lossless-buffers=auto lossless-traffic-class=3

# Profiles: RoCE data in TC3, congestion notification packets (CNP) in TC6.
/interface ethernet switch qos profile add name=roce-lossless pcp=3 dscp=0 traffic-class=3 automap=no
/interface ethernet switch qos profile add name=roce-cnp pcp=6 dscp=48 traffic-class=6 automap=no

# Map DSCP 26 (spark-ds41f on a switch marks traffic class 106 = DSCP 26 + ECT(0)) to TC3, DSCP 48 to TC6.
/interface ethernet switch qos map add name=roce-lossless
/interface ethernet switch qos map ip add map=roce-lossless dscp=26 profile=roce-lossless
# Optional: an engine that sends traffic class 0 needs DSCP 0 in the lossless class as well.
# The measured site used this instead of the line above; it makes ALL unmarked traffic on
# these ports lossless:
#   /interface ethernet switch qos map ip add map=roce-lossless dscp=0,26 profile=roce-lossless
/interface ethernet switch qos map ip add map=roce-lossless dscp=48 profile=roce-cnp

# ECN marking on queue 3 (no WRED drops).
/interface ethernet switch qos tx-manager queue set [find tx-manager=default traffic-class=3] wred=no ecn=yes

# PFC receive and transmit on TC3.
/interface ethernet switch qos priority-flow-control add name=roce-lossless rx=yes tx=yes traffic-class=3

# Apply to the fabric ports only: keep the packet's DSCP, use the map and PFC above.
/interface ethernet switch qos port set [find name="<FABRIC_PORT_1>"] map=roce-lossless trust-l3=keep pfc=roce-lossless
/interface ethernet switch qos port set [find name="<FABRIC_PORT_2>"] map=roce-lossless trust-l3=keep pfc=roce-lossless
/interface ethernet switch qos port set [find name="<FABRIC_PORT_3>"] map=roce-lossless trust-l3=keep pfc=roce-lossless
/interface ethernet switch qos port set [find name="<FABRIC_PORT_4>"] map=roce-lossless trust-l3=keep pfc=roce-lossless

# Receive PFC needs an explicit queue-3 rate. Use the real wire rate: 7.23.5 accepts 200G
# (7.20.8 refused anything above 100G).
/interface ethernet switch qos port set [find name="<FABRIC_PORT_1>"] egress-rate-queue3=200G
/interface ethernet switch qos port set [find name="<FABRIC_PORT_2>"] egress-rate-queue3=200G
/interface ethernet switch qos port set [find name="<FABRIC_PORT_3>"] egress-rate-queue3=200G
/interface ethernet switch qos port set [find name="<FABRIC_PORT_4>"] egress-rate-queue3=200G

# Read back. Expect no "requires rate" warnings on the PFC ports, then confirm with counters
# (pause frames, queue drops) while a collective runs: accepted settings are not proof.
/interface ethernet switch qos settings print
/interface ethernet switch qos priority-flow-control print detail without-paging
/interface ethernet switch qos port print detail without-paging where name~"<FABRIC_PORT_PATTERN>"
