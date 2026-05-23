#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/paddlehelix_chemvl/bin/python}"
RUNSEED_START="${RUNSEED_START:-1}"
RUNSEED_END="${RUNSEED_END:-3}"
MAX_EPOCH="${MAX_EPOCH:-100}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LOADER_WORKERS="${LOADER_WORKERS:-1}"
export OMP_NUM_THREADS="${GEM_OMP_NUM_THREADS:-1}"
EXTRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then EXTRA_ARGS+=(--dry-run); fi
if [[ "${SPLIT_ONLY:-0}" == "1" ]]; then EXTRA_ARGS+=(--split-only --capture-output); fi
if [[ "${NO_SKIP:-0}" == "1" ]]; then EXTRA_ARGS+=(--no-skip-existing); fi

"$PYTHON" "$SCRIPT_DIR/batch_run.py" \
  --split random_scaffold \
  --task-type classification \
  --dataset-list "$SCRIPT_DIR/dataset_list_moleculenet_cls6.txt" \
  --runseed-start "$RUNSEED_START" \
  --runseed-end "$RUNSEED_END" \
  --max-epoch "$MAX_EPOCH" \
  --num-workers "$NUM_WORKERS" \
  --loader-workers "$LOADER_WORKERS" \
  "${EXTRA_ARGS[@]}"

"$PYTHON" "$SCRIPT_DIR/batch_run.py" \
  --split random_scaffold \
  --task-type regression \
  --dataset-list "$SCRIPT_DIR/dataset_list_moleculenet_reg4.txt" \
  --runseed-start "$RUNSEED_START" \
  --runseed-end "$RUNSEED_END" \
  --max-epoch "$MAX_EPOCH" \
  --num-workers "$NUM_WORKERS" \
  --loader-workers "$LOADER_WORKERS" \
  "${EXTRA_ARGS[@]}"
