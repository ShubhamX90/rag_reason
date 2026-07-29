#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

head -n 8 data/messages/train_stagewise_e2e_messages.jsonl > data/messages/smoke_train_stagewise_e2e.jsonl
head -n 4 data/messages/val_stagewise_e2e_messages.jsonl > data/messages/smoke_val_stagewise_e2e.jsonl
head -n 2 data/messages/val_stagewise_e2e_messages.jsonl > data/messages/smoke_gen_val_stagewise_e2e.jsonl

echo "Created:"
echo "  data/messages/smoke_train_stagewise_e2e.jsonl"
echo "  data/messages/smoke_val_stagewise_e2e.jsonl"
echo "  data/messages/smoke_gen_val_stagewise_e2e.jsonl"
