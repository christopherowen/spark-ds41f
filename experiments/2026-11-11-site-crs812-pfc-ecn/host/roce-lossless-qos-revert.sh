#!/bin/sh
# Undo roce-lossless-qos.sh. Run as root with serving stopped.
set -eu
systemctl disable --now roce-lossless-qos.service || true
rm -f /etc/NetworkManager/dispatcher.d/91-roce-lossless-qos \
      /etc/systemd/system/roce-lossless-qos.service \
      /usr/local/sbin/roce-lossless-qos
systemctl daemon-reload
for iface in enp1s0f0np0 enP2p1s0f0np0; do
  mlnx_qos -i "$iface" --pfc=0,0,0,0,0,0,0,0
  dcb app del dev "$iface" dscp-prio 26:3 48:6 || true
  mlnx_qos -i "$iface" --trust=pcp
  ethtool -A "$iface" rx on tx on
done
