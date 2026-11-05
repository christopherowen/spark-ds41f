#!/bin/bash
# MemAvailable on every rank every 5 s while a run goes. Run on the head:
#   bash memwatch.sh <out.tsv> <stop-file> <rank1-ssh-host> <rank2-ssh-host> <rank3-ssh-host>
# Warn-only: a rank below 6 GiB is logged as WARN, nothing is stopped. Ends when <stop-file> exists.
# Published files name the ranks (rank0 = head).
out=$1; stop=$2; shift 2
printf 'utc\tnode\tmemavail_gib\n' > "$out"
while [ ! -e "$stop" ]; do
  t=$(date -u +%T)
  r=0
  for h in localhost "$@"; do
    if [ "$h" = localhost ]; then kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    else kb=$(ssh -o BatchMode=yes -o ConnectTimeout=3 "$h" "awk '/MemAvailable/ {print \$2}' /proc/meminfo" 2>/dev/null); fi
    n=rank$r; r=$((r + 1))
    [ -n "$kb" ] || { printf '%s\t%s\tNA\n' "$t" "$n" >> "$out"; continue; }
    g=$(awk -v k="$kb" 'BEGIN {printf "%.2f", k/1048576}')
    printf '%s\t%s\t%s\n' "$t" "$n" "$g" >> "$out"
    awk -v g="$g" 'BEGIN {exit !(g < 6)}' && printf '%s\t%s\tWARN below 6 GiB: %s\n' "$t" "$n" "$g" >> "$out"
  done
  sleep 5
done
echo "MEMWATCH_END" >> "$out"
