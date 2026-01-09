#!/bin/bash
set -e

CONFIG_PATH="config.yaml"
EXP_DIR="experiments"
RUN_NAME=$(date +"%Y%m%d_%H%M%S")

RUN_DIR="${EXP_DIR}/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/train.log"

mkdir -p "${RUN_DIR}"

echo "Run name: ${RUN_NAME}"
echo "Log: ${LOG_FILE}"

export RUN_NAME="${RUN_NAME}"

PYTHONDONTWRITEBYTECODE=1 python src/main.py 2>&1 | tee "${LOG_FILE}"
