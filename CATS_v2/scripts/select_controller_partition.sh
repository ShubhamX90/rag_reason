#!/usr/bin/env bash
set -euo pipefail

NEED_CPUS="${1:-4}"
PRIMARY_PARTITION="${PRIMARY_CONTROLLER_PARTITION:-compute}"
FORCE_PARTITION="${FORCE_CONTROLLER_PARTITION:-}"

partition_qos() {
  case "$1" in
    compute) echo "cpulimit" ;;
    *) echo "" ;;
  esac
}

partition_min_cpus() {
  echo "0"
}

emit_selection() {
  local partition="$1"
  local reason="$2"
  local request_cpus=""

  request_cpus="$NEED_CPUS"
  local min_cpus=""
  min_cpus="$(partition_min_cpus "$partition")"
  if [[ "$min_cpus" =~ ^[0-9]+$ ]] && (( request_cpus < min_cpus )); then
    request_cpus="$min_cpus"
  fi

  echo "partition=$partition"
  echo "qos=$(partition_qos "$partition")"
  echo "request_cpus=$request_cpus"
  echo "reason=$reason"
}

if [[ -n "$FORCE_PARTITION" ]]; then
  emit_selection "$FORCE_PARTITION" "forced"
  exit 0
fi

emit_selection "$PRIMARY_PARTITION" "compute_only"
