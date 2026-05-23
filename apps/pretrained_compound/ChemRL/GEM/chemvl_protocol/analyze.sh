#!/usr/bin/env bash
set -euo pipefail

CHEMVL_REPO="${CHEMVL_REPO:-/root/autodl-tmp/ChemVL-private}"
DATA_ROOT="${CHEMVL_DATA_ROOT:-${CHEMVL_REPO}/chemvl-data}"
PYTHON="${PYTHON:-/root/miniconda3/envs/paddlehelix_chemvl/bin/python}"
EXP_NAME="${EXP_NAME:-gem_under_chemvl}"
RESULT_ROOT="${RESULT_ROOT:-${DATA_ROOT}/results/moleculenet/${EXP_NAME}}"

"$PYTHON" "${CHEMVL_REPO}/scripts/moleculeace_batch_analyze.py" \
  --root "$RESULT_ROOT" \
  --out-stem gem_under_chemvl \
  "$@"
