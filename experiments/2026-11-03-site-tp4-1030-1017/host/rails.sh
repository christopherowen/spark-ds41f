#!/bin/bash
# ib_write_bw, 1 MiB, 5 s, server <- client, per fabric rail. Run on the head. Arguments: rails as server_host:server_ip:client_host, e.g.
#   bash rails.sh rank3:198.51.100.1:rank2 rank3:198.51.100.129:localhost   (ssh host names, rail addresses)
# The RDMA device on each side is the one whose netdev holds the rail address (server) or the same subnet (client).
dev_for() {  # host, address prefix (full address for the server, subnet for the client)
  local cmd='ifc=$(ip -4 -br addr | awk -v p="'"$2"'" "index(\$3, p) == 1 {print \$1}"); for d in /sys/class/infiniband/*; do [ -e "$d/device/net/$ifc" ] && basename "$d"; done'
  if [ "$1" = localhost ]; then bash -c "$cmd"; else ssh -o BatchMode=yes "$1" "$cmd"; fi
}
run_on() { if [ "$1" = localhost ]; then shift; bash -c "$*"; else local h=$1; shift; ssh -o BatchMode=yes "$h" "$*"; fi; }
for rail in "$@"; do
  IFS=: read -r shost sip chost <<<"$rail"
  subnet=${sip%.*}.
  sdev=$(dev_for "$shost" "$sip/")
  cdev=$(dev_for "$chost" "$subnet")
  run_on "$shost" "timeout 40 ib_write_bw -d $sdev -s 1048576 -D 5 --report_gbits -F -R >/dev/null 2>&1" &
  sleep 2
  bw=$(run_on "$chost" "timeout 30 ib_write_bw -d $cdev -s 1048576 -D 5 --report_gbits -F -R $sip 2>&1" |
    awk '$1 == 1048576 {print $4}')
  wait
  echo "rail $subnet* $shost($sdev) <- $chost($cdev): ${bw:-FAIL} Gb/s"
done
