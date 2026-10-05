#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

LEFT_DATASET="${LEFT_DATASET:-${MODEL_DIR}/WS_L_1M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt}"
RIGHT_DATASET="${RIGHT_DATASET:-${MODEL_DIR}/WS_R_1M_R1_5_pT250/datasets_AllFrame/dataset_combined.pt}"
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/WS_U_10M_R1_5_pT250/training_down_60epoch_64embed/model_final.torch}"
TASK="${TASK:-down}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_DIR}/eval_test}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-auto}"
PYTHON="${PYTHON:-python}"

exec "${PYTHON}" "${MODEL_DIR}/evaluate.py" \
    --left-dataset "${LEFT_DATASET}" \
    --right-dataset "${RIGHT_DATASET}" \
    --model "${MODEL_PATH}" \
    --task "${TASK}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}"
