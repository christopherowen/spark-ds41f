#!/usr/bin/env bash
set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# bin/spark passes the settings from the resolved configuration (an arm overlay or a
# recipe has no container or host fields of its own).
if (($# != 6 && $# != 7)); then
  echo "usage: $0 <container> <startup|steady> <threshold-gib> <interval-seconds> <wait-seconds> <running|prelaunch> [warn-gib]" >&2
  exit 2
fi
name=$1
phase=$2
guard_gib=$3
interval=$4
wait_seconds=$5
mode=$6
warn_gib=${7:-0}
number='^[0-9]+([.][0-9]+)?$'
if [[ ! "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "memguard container name is not a container name: $name" >&2
  exit 2
fi
case "$phase" in
  startup | steady) ;;
  *)
    echo "memguard phase must be startup or steady" >&2
    exit 2
    ;;
esac
for value in "$guard_gib" "$interval" "$wait_seconds"; do
  if [[ ! "$value" =~ $number ]]; then
    echo "memguard threshold, interval and wait must be non-negative numbers: $value" >&2
    exit 2
  fi
done
if [[ ! "$warn_gib" =~ ^[0-9]+$ ]]; then
  echo "memguard warn line must be a non-negative integer: $warn_gib" >&2
  exit 2
fi
# The startup guard runs at the launch line, above the steady one: a warn line belongs
# to the steady guard only.
if ((warn_gib > 0)) && [[ "$phase" != "steady" ]]; then
  echo "the memguard warn line is valid only for the steady memguard" >&2
  exit 2
fi
unit="${name}-memguard.service"

case "$mode" in
  running)
    if [[ "$(docker inspect --format '{{.State.Running}}' "$name" 2>/dev/null || true)" != "true" ]]; then
      echo "refusing to arm memguard: container $name is not running" >&2
      exit 1
    fi
    wait_seconds=0
    ;;
  prelaunch)
    if [[ "$phase" != "startup" ]]; then
      echo "prelaunch mode is valid only for the startup memguard" >&2
      exit 2
    fi
    ;;
  *)
    echo "memguard mode must be running or prelaunch" >&2
    exit 2
    ;;
esac

runtime_dir=${XDG_RUNTIME_DIR:-/tmp}
if [[ ! -d "$runtime_dir" || ! -w "$runtime_dir" ]]; then
  runtime_dir=/tmp
fi
ready_file=$(mktemp "$runtime_dir/${name}-memguard.XXXXXX")
rm -f -- "$ready_file"

# On heterogeneous hosts (e.g. GB10: Cluster 0 efficiency cores 0-4 in 8 MiB L3 vs
# Cluster 1 serving cores 15-19 in 16 MiB L3), confine memguard's periodic docker/proc
# checks to the smallest-L3 cluster's efficiency cores so Cluster 1 L3 is undisturbed.
resolve_housekeeping_cpus() {
  local root=${MEMGUARD_SYSFS_CPU_ROOT:-/sys/devices/system/cpu}
  [[ -d "$root" ]] || return 0
  python3 - "$root" 2>/dev/null <<'PY' || true
import os, sys
root = sys.argv[1]
def read_int(p):
    try:
        t = open(p, encoding="utf-8").read().strip().upper()
        if t.endswith("K"): return int(t[:-1])
        if t.endswith("M"): return int(t[:-1]) * 1024
        return int(t)
    except Exception:
        return -1
def parse_list(s):
    out = []
    for part in s.strip().split(","):
        if not part: continue
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out
cpus = sorted(int(n[3:]) for n in os.listdir(root) if n.startswith("cpu") and n[3:].isdigit())
scores = {c: read_int(f"{root}/cpu{c}/cpufreq/cpuinfo_max_freq") for c in cpus}
if not any(v > 0 for v in scores.values()):
    scores = {c: read_int(f"{root}/cpu{c}/cpu_capacity") for c in cpus}
scores = {c: v for c, v in scores.items() if v > 0}
if not scores or min(scores.values()) == max(scores.values()):
    sys.exit(0)
mid = (min(scores.values()) + max(scores.values())) // 2
eff = {c for c, v in scores.items() if v <= mid}
if not eff:
    sys.exit(0)
clusters = []
for c in cpus:
    try:
        grp = tuple(parse_list(open(f"{root}/cpu{c}/cache/index3/shared_cpu_list", encoding="utf-8").read()))
    except Exception:
        continue
    kib = read_int(f"{root}/cpu{c}/cache/index3/size")
    if grp and (kib, grp) not in clusters:
        clusters.append((kib if kib > 0 else 1 << 30, grp))
if clusters:
    clusters.sort(key=lambda x: (x[0], x[1][0]))
    for _, grp in clusters:
        cand = sorted(c for c in grp if c in eff)
        if cand:
            print(",".join(map(str, cand)))
            sys.exit(0)
print(",".join(map(str, sorted(eff))))
PY
}

cpu_prop=()
housekeeping_cpus=$(resolve_housekeeping_cpus)
if [[ -n "$housekeeping_cpus" ]]; then
  cpu_prop=(--property="AllowedCPUs=$housekeeping_cpus")
fi

# A collected transient unit may still have a failed record from a prior stop.
sudo -n install -d -m 0755 -o "$(id -un)" /var/log/spark-memguard >/dev/null 2>&1 || true
sudo -n systemctl stop "$unit" >/dev/null 2>&1 || true
sudo -n systemctl reset-failed "$unit" >/dev/null 2>&1 || true
sudo -n systemd-run --quiet \
  --unit="${name}-memguard" \
  --collect \
  --property=Type=simple \
  --property="User=$(id -un)" \
  --property="WorkingDirectory=$repo_root" \
  --property=OOMScoreAdjust=-1000 \
  --property=MemoryMin=64M \
  --property=CPUWeight=1000 \
  ${cpu_prop[@]+"${cpu_prop[@]}"} \
  --setenv="THRESHOLD_GIB=$guard_gib" \
  --setenv="INTERVAL=$interval" \
  --setenv="CONTAINER_NAME=$name" \
  --setenv="MEMGUARD_PHASE=$phase" \
  --setenv="WAIT_FOR_CONTAINER_SECONDS=$wait_seconds" \
  --setenv="MEMGUARD_WARN_GIB=$warn_gib" \
  --setenv="READY_FILE=$ready_file" \
  "$repo_root/scripts/memguard.sh"

for _ in {1..50}; do
  if [[ -f "$ready_file" ]]; then
    rm -f -- "$ready_file"
    warn_text="warn line ${warn_gib} GiB"
    ((warn_gib > 0)) || warn_text="no warn line"
    echo "$phase memguard active for $name (threshold ${guard_gib} GiB, interval ${interval}s, $warn_text, mode $mode)"
    exit 0
  fi
  if ! sudo -n systemctl is-active --quiet "$unit"; then
    echo "memguard exited before completing its initial protected sample" >&2
    exit 1
  fi
  sleep 0.1
done

sudo -n systemctl stop "$unit" >/dev/null 2>&1 || true
echo "memguard did not confirm its initial protected sample" >&2
exit 1
