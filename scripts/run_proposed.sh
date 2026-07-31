#!/bin/bash
# Run training for proposed AquaNet v3 model on cleaned dataset

set -e

echo "=========================================================================="
echo " Training Proposed Model: AquaNet v3 (DenseNet + MSRB/CSAB + Soft Dual-Head)"
echo "=========================================================================="

python train.py --model aquanet_v3 --epochs 50 --batch-size 32 --lr 0.001

echo "Evaluating AquaNet v3 across all 3 datasets (D1, D2, D3)..."
python test.py --model aquanet_v3 --checkpoint checkpoints/aquanet_v3_best.pth --dataset all

echo "AquaNet v3 execution complete!"
