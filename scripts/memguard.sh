#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_NAME:?CONTAINER_NAME is required}"
: "${THRESHOLD_GIB:?THRESHOLD_GIB is required}"
: "${INTERVAL:?INTERVAL is required}"
: "${MEMGUARD_PHASE:=steady}"
: "${WAIT_FOR_CONTAINER_SECONDS:=0}"
: "${MEMGUARD_WARN_GIB:=0}"
: "${READY_FILE:=}"
: "${MEMGUARD_EVIDENCE_DIR:=/var/log/spark-memguard}"
: "${MEMGUARD_PROC_DIR:=/proc}"
: "${MEMGUARD_MEMINFO_FILE:=$MEMGUARD_PROC_DIR/meminfo}"

if ! [[ "$THRESHOLD_GIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "THRESHOLD_GIB must be a positive integer" >&2
  exit 2
fi
if ! [[ "$INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$INTERVAL" =~ ^0+([.]0+)?$ ]]; then
  echo "INTERVAL must be a positive number" >&2
  exit 2
fi
if ! [[ "$WAIT_FOR_CONTAINER_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "WAIT_FOR_CONTAINER_SECONDS must be a non-negative integer" >&2
  exit 2
fi
# The warn line only reports: it never delays or replaces the kill below THRESHOLD_GIB.
if ! [[ "$MEMGUARD_WARN_GIB" =~ ^[0-9]+$ ]]; then
  echo "MEMGUARD_WARN_GIB must be a non-negative integer" >&2
  exit 2
fi
if ((MEMGUARD_WARN_GIB > 0 && MEMGUARD_WARN_GIB <= THRESHOLD_GIB)); then
  echo "MEMGUARD_WARN_GIB must exceed THRESHOLD_GIB" >&2
  exit 2
fi

threshold_kib=$((THRESHOLD_GIB * 1024 * 1024))
warn_kib=$((MEMGUARD_WARN_GIB * 1024 * 1024))
deadline=$((SECONDS + WAIT_FOR_CONTAINER_SECONDS))
observed_running=false
ready=false
warned=false
previous_kib=
previous_at=
container_pid=
container_starttime=

# Read field 22 (starttime) from /proc/<pid>/stat with pure bash builtins (no forks).
read_pid_starttime() {
  local pid=$1 stat_line after_comm
  local -a fields
  pid_starttime=
  [[ -r "$MEMGUARD_PROC_DIR/$pid/stat" ]] || return 1
  IFS= read -r stat_line <"$MEMGUARD_PROC_DIR/$pid/stat" 2>/dev/null || return 1
  after_comm=${stat_line##*) }
  [[ "$after_comm" != "$stat_line" ]] || return 1
  read -r -a fields <<<"$after_comm"
  [[ -n "${fields[19]:-}" ]] || return 1
  pid_starttime=${fields[19]}
}

# Read MemAvailable (KiB) directly from /proc/meminfo with pure bash builtins (no forks).
read_mem_available_kib() {
  local key value rest
  available_kib=
  [[ -r "$MEMGUARD_MEMINFO_FILE" ]] || return 1
  while read -r key value rest; do
    if [[ "$key" == "MemAvailable:" ]]; then
      available_kib=$value
      return 0
    fi
  done <"$MEMGUARD_MEMINFO_FILE"
  return 0
}

# After a kill, what `docker rm -f` would otherwise destroy: the breach's meminfo, the
# largest host processes, the container's state and its log. Never delays the kill.
save_evidence() {
  local meminfo=$1 stamp dir
  stamp="${CONTAINER_NAME}-$(date -u +%Y%m%dT%H%M%SZ)"
  dir="$MEMGUARD_EVIDENCE_DIR/$stamp"
  if ! mkdir -p "$dir" 2>/dev/null; then
    dir="${XDG_STATE_HOME:-${HOME:-/tmp}/.local/state}/spark-memguard/$stamp"
    if ! mkdir -p "$dir" 2>/dev/null; then
      dir="${XDG_RUNTIME_DIR:-/tmp}/spark-memguard/$stamp"
      if ! mkdir -p "$dir" 2>/dev/null; then
        logger -t spark-memguard "cannot create evidence directory for the kill of $CONTAINER_NAME" || true
        return 0
      fi
    fi
  fi
  printf '%s\n' "$meminfo" >"$dir/meminfo.txt" 2>/dev/null || true
  ps -eo pid,rss,vsz,comm --sort=-rss 2>/dev/null | head -n 25 >"$dir/ps-rss.txt" || true
  timeout 30 docker inspect "$CONTAINER_NAME" >"$dir/inspect.json" 2>/dev/null || true
  timeout 60 docker logs --timestamps --tail 2000 "$CONTAINER_NAME" >"$dir/container.log" 2>&1 || true
  logger -t spark-memguard "evidence for the kill of $CONTAINER_NAME: $dir" || true
}

while true; do
  inspectable=true
  running=
  if [[ -n "$container_pid" ]] && read_pid_starttime "$container_pid" && [[ "$pid_starttime" == "$container_starttime" ]]; then
    running=true
  else
    container_pid=
    container_starttime=
    if ! inspect_out=$(docker inspect --format '{{.State.Running}} {{.State.Pid}}' "$CONTAINER_NAME" 2>/dev/null); then
      inspectable=false
      running=
    else
      inspected_pid=
      read -r running inspected_pid <<<"$inspect_out"
      if [[ "$running" == "true" && "$inspected_pid" =~ ^[1-9][0-9]*$ ]] && read_pid_starttime "$inspected_pid"; then
        container_pid=$inspected_pid
        container_starttime=$pid_starttime
      fi
    fi
  fi
  if [[ "$inspectable" == "false" && "$observed_running" == "true" ]]; then
    logger -t spark-memguard \
      "cannot inspect $CONTAINER_NAME after startup during $MEMGUARD_PHASE; failing closed"
    docker kill --signal KILL "$CONTAINER_NAME" >/dev/null 2>&1 || true
    exit 1
  fi
  if [[ "$running" == "true" ]]; then
    observed_running=true
  elif [[ "$observed_running" == "true" ]]; then
    exit 0
  elif ((SECONDS >= deadline)); then
    logger -t spark-memguard \
      "$CONTAINER_NAME did not start within ${WAIT_FOR_CONTAINER_SECONDS}s; leaving no unbounded watcher"
    exit 1
  fi

  available_kib=
  for ((read_try = 0; read_try < 3; read_try++)); do
    read_mem_available_kib || available_kib=
    if [[ -n "$available_kib" && "$available_kib" =~ ^[0-9]+$ ]]; then
      break
    fi
    if ((read_try < 2)); then
      sleep 0.05
    fi
  done
  if [[ -z "$available_kib" || ! "$available_kib" =~ ^[0-9]+$ ]]; then
    logger -t spark-memguard \
      "cannot read MemAvailable during $MEMGUARD_PHASE; failing closed"
    if [[ "$running" == "true" ]]; then
      docker kill --signal KILL "$CONTAINER_NAME" >/dev/null
    fi
    exit 1
  fi
  # The guard is a fixed line: the first sample below it ends the container.
  if ((available_kib < threshold_kib)); then
    breach="MemAvailable ${available_kib} KiB below ${threshold_kib} KiB (short by $((threshold_kib - available_kib)) KiB)"
    if ((warn_kib > 0)); then
      breach+="; warn line ${warn_kib} KiB"
    fi
    if [[ -n "$previous_kib" ]]; then
      breach+="; previous sample ${previous_kib} KiB $((SECONDS - previous_at))s earlier (down $((previous_kib - available_kib)) KiB)"
    fi
    # One interval can carry MemAvailable past both lines; say so rather than leave a
    # kill with no warning unexplained.
    if ((warn_kib > 0)) && [[ "$warned" == "false" ]]; then
      if [[ -n "$previous_kib" ]]; then
        breach+="; no warning: it fell from above the warn line to below the kill line between two samples"
      else
        breach+="; no warning: the first sample was below the kill line"
      fi
    fi
    if [[ "$running" == "true" ]]; then
      { meminfo=$(<"$MEMGUARD_MEMINFO_FILE"); } 2>/dev/null || meminfo=
      logger -t spark-memguard "killing $CONTAINER_NAME during $MEMGUARD_PHASE: $breach"
      docker kill --signal KILL "$CONTAINER_NAME" >/dev/null
      save_evidence "$meminfo"
    else
      logger -t spark-memguard "refusing to wait for $CONTAINER_NAME during $MEMGUARD_PHASE: $breach"
    fi
    exit 1
  fi
  if ((warn_kib > 0 && available_kib < warn_kib)); then
    if [[ "$warned" == "false" ]]; then
      logger -t spark-memguard \
        "low headroom on $CONTAINER_NAME during $MEMGUARD_PHASE: MemAvailable ${available_kib} KiB below the ${warn_kib} KiB warn line (kill line ${threshold_kib} KiB)" ||
        true
      warned=true
    fi
  else
    warned=false
  fi
  previous_kib=$available_kib
  previous_at=$SECONDS

  if [[ "$ready" == "false" ]]; then
    if [[ -n "$READY_FILE" ]]; then
      : >"$READY_FILE"
    fi
    ready=true
  fi
  sleep "$INTERVAL"
done
