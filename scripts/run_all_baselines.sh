#!/bin/bash
# High-Throughput Baseline & Proposed Models Benchmark Script (NVIDIA A100 GPU)

set -e

PYTHON_ENV="/workspace/miniconda3/envs/vision/bin/python"
MODELS=("aquanet_v1" "aquanet_v2" "aquanet_v3" "resnet50" "densenet121" "efficientnet_b0" "mobilenetv2" "vit_tiny" "swin_tiny")

BATCH_SIZE=64
EPOCHS=30

for model in "${MODELS[@]}"; do
    echo "=========================================================================="
    echo " Training Model: $model | Batch Size: $BATCH_SIZE | Epochs: $EPOCHS"
    echo "=========================================================================="
    "$PYTHON_ENV" train.py --model "$model" --epochs $EPOCHS --batch-size $BATCH_SIZE --lr 0.001
done

echo "All baseline and proposed model training completed!"
