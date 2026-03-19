#!/usr/bin/env bash
# Launch training detached from the shell, logging to the output directory.
# Usage:
#   ./train/run.sh                              # default config
#   ./train/run.sh --config path/to/other.yaml  # custom config
#   ./train/run.sh --resume                     # resume from checkpoint

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/train_config.yaml"

# Parse --config if provided (need it to derive output dir)
TRAIN_ARGS=("$@")
for i in "${!TRAIN_ARGS[@]}"; do
    if [[ "${TRAIN_ARGS[$i]}" == "--config" && -n "${TRAIN_ARGS[$((i+1))]:-}" ]]; then
        CONFIG="${TRAIN_ARGS[$((i+1))]}"
    fi
done

# Derive the output directory from the YAML config (same logic as train.py)
OUTPUT_DIR=$(python3 -c "
import yaml, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
version_tag = f'-v{cfg[\"version\"]:02d}'
short_name = cfg['model_id'].split('/')[-1]
print(f'outputs/train/{short_name}{version_tag}')
")

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOGDIR="${REPO_ROOT}/logs"
mkdir -p "${LOGDIR}"

RUN_NAME="$(basename ${OUTPUT_DIR})"
LOGFILE="${LOGDIR}/${RUN_NAME}.log"
PIDFILE="${LOGDIR}/${RUN_NAME}.pid"

echo "=== Starting training ==="
echo "  Config:   ${CONFIG}"
echo "  Output:   ${OUTPUT_DIR}"
echo "  Log file: ${LOGFILE}"
echo ""

nohup python3 "${SCRIPT_DIR}/train.py" "$@" > "${LOGFILE}" 2>&1 &
PID=$!

echo "Training started in background (PID: ${PID})"
echo "  Tail logs:  tail -f ${LOGFILE}"
echo "  Stop:       kill ${PID}"
echo "${PID}" > "${PIDFILE}"
