#!/bin/bash
# Asynchronous Background Training Runner (Nohup Protected)
# Runs AquaNet-VLM (and all models) in background so execution continues uninterrupted.

set -e

mkdir -p logs checkpoints results

LOG_FILE="logs/nohup_aquanet_vlm_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================================================="
echo " Launching Background Training for AquaNet-VLM"
echo " Log file: $LOG_FILE"
echo "=========================================================================="

nohup env PYTHONUNBUFFERED=1 /workspace/miniconda3/envs/vision/bin/python train.py \
    --model aquanet_vlm \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training process launched in background with PID: $PID"
echo "To monitor progress, run: tail -f $LOG_FILE"
