#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${LOG_ROOT:-/root/autodl-tmp/ChemVL-private/chemvl-data/results/moleculenet/gem_under_chemvl_logs}"
mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/ab_${STAMP}.log"
PID_FILE="$LOG_ROOT/ab_${STAMP}.pid"

setsid bash -c 'echo "$BASHPID" > "$1"; export GEM_OMP_NUM_THREADS="${GEM_OMP_NUM_THREADS:-1}"; bash "$2/run_moleculenet_scaffold.sh" && bash "$2/run_moleculenet_random_scaffold.sh"' _ "$PID_FILE" "$SCRIPT_DIR" > "$LOG_FILE" 2>&1 < /dev/null &
echo "ab pidfile=$PID_FILE log=$LOG_FILE"
