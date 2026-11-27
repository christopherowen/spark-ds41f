#!/bin/sh
# Host side of the lossless RoCE class, for each Spark on a switched fabric. Run as root.
# Port 0, both PCIe paths (docs/fabric-conventions.md). Changes only DCB/QoS on those two
# interfaces: no addresses, routes, MTU or other NICs.
#   - trust DSCP, map DSCP 26 -> priority 3 and DSCP 48 (CNP) -> priority 6
#   - PFC on priority 3 only
#   - 802.3x global pause off (PFC replaces it)
# Installs a oneshot systemd unit and a NetworkManager dispatcher hook so the state survives
# reboots and link flaps (dcb state does not persist by itself).
set -eu

cat > /usr/local/sbin/roce-lossless-qos <<'HELPER'
#!/bin/sh
set -eu
exec 9>/run/lock/roce-lossless-qos.lock
flock -w 30 9
if [ "$#" -eq 0 ]; then set -- enp1s0f0np0 enP2p1s0f0np0; fi
for iface in "$@"; do
  case "$iface" in enp1s0f0np0|enP2p1s0f0np0) ;; *) echo "unexpected interface: $iface" >&2; exit 2 ;; esac
  test -e "/sys/class/net/$iface"
  ethtool -A "$iface" rx off tx off
  mlnx_qos -i "$iface" --trust=dscp
  # Optional, only for an engine that sends traffic class 0 (the measured site used 0:3 26:3 48:6):
  #   dcb app replace dev "$iface" dscp-prio 0:3 26:3 48:6
  dcb app replace dev "$iface" dscp-prio 26:3 48:6
  mlnx_qos -i "$iface" --pfc=0,0,0,1,0,0,0,0
done
HELPER
chmod 0755 /usr/local/sbin/roce-lossless-qos

cat > /etc/systemd/system/roce-lossless-qos.service <<'UNIT'
[Unit]
Description=RoCE DSCP trust and priority flow control on the fabric port
Wants=network-online.target
After=network-online.target NetworkManager-wait-online.service
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/roce-lossless-qos
RemainAfterExit=yes
TimeoutStartSec=90
[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/NetworkManager/dispatcher.d/91-roce-lossless-qos <<'DISPATCH'
#!/bin/sh
case "$1:$2" in
  enp1s0f0np0:up|enp1s0f0np0:reapply|enP2p1s0f0np0:up|enP2p1s0f0np0:reapply)
    /usr/local/sbin/roce-lossless-qos "$1" ;;
esac
DISPATCH
chmod 0755 /etc/NetworkManager/dispatcher.d/91-roce-lossless-qos

systemctl daemon-reload
systemctl enable --now roce-lossless-qos.service

# Read back.
for iface in enp1s0f0np0 enP2p1s0f0np0; do
  dcb app show dev "$iface"
  mlnx_qos -i "$iface"
done
