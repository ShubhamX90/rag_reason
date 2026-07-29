#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 JOB_ID PORT" >&2
  exit 1
fi

JOB_ID="$1"
PORT="$2"

node="$(squeue -h -j "$JOB_ID" -o "%N" 2>/dev/null | head -n1 || true)"
if [[ -z "$node" || "$node" == "(null)" || "$node" == "n/a" ]]; then
  node="$(sacct -X -j "$JOB_ID" --parsable2 --format=JobIDRaw,NodeList -n 2>/dev/null | awk -F'|' -v id="$JOB_ID" '$1 == id && $2 != "" {print $2; exit}')"
fi

if [[ -z "$node" ]]; then
  echo "Could not resolve node for job $JOB_ID" >&2
  exit 1
fi

host="$(scontrol show hostnames "$node" 2>/dev/null | head -n1 || true)"
if [[ -z "$host" ]]; then
  host="$node"
fi

node_addr="$(scontrol show node "$host" 2>/dev/null | sed -n 's/.*NodeAddr=\([^ ]*\).*/\1/p' | head -n1 || true)"
if [[ -n "$node_addr" ]]; then
  host="$node_addr"
fi

printf 'http://%s:%s/v1\n' "$host" "$PORT"
