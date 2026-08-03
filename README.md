# AquaNet: Controlled Visual Water-Condition Benchmarking

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/pytorch-2.6.0-orange.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/cuda-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)

PyTorch research code and completed evidence package for seven-class visual water-surface condition recognition (Clean, Algae, Debris, Foam, Oil, Turbid, Uncertain). RGB appearance is not a physicochemical water-quality measurement.

## Current result and manuscript

Phase 4 is complete: 222 successful training runs plus calibration, abstention, corruption, transfer, and explanation evaluations. The central result is a protocol effect: forcing unlike pretrained and custom models to share hyperparameter values reversed their ranking. With an equal per-model tuning budget, AquaNet-full ranks ninth of ten and its added MSRB/CSAB modules are null-to-negative.

The current manuscript is [`paper_final/main.tex`](paper_final/main.tex), with compiled PDF at [`paper_final/main.pdf`](paper_final/main.pdf). The immutable narrative and full result tables are in [`AQUANET_Q4_PLAN.md`](AQUANET_Q4_PLAN.md); `paper_v1`, `paper_v2`, and `paper_v3` are historical drafts only.

---

## 📁 Repository Directory Layout

```
aquanet/
├── train.py                            # Unified CLI training script
├── test.py                             # Unified evaluation script across D1, D2, D3
├── inference.py                        # Single image & batch inference API
├── configs/
│   ├── base_config.yaml                # Hyperparameters & paths
│   └── models_config.yaml              # Per-model configuration
├── data/
│   ├── cleaned_water_dataset/          # D1: Main benchmark set (2,799 unique images)
│   ├── cleaned_scrapper_finetune/      # D2: Real video fine-tune set (213 unique images)
│   └── cleaned_scrapper_unseen_test/   # D3: Zero-shot OOD test set (146 unique images)
├── dataset/
│   ├── water_dataset.py                # Dataset class for 7-class & dual-head
│   └── transforms.py                   # Data augmentation pipelines
├── models/
│   ├── classical/                      # Classical ML (Random Forest, KNN, SVM)
│   ├── deep_learning/                  # DL Baselines (ResNet50, DenseNet121, EfficientNet, MobileNet)
│   ├── transformers/                   # ViT-Tiny & Swin-Tiny
│   └── proposed/                       # AquaNet v3 (DenseNet + MSRB/CSAB + Soft Dual-Head)
├── utils/
│   ├── logger.py                       # Logging setup
│   ├── metrics.py                      # Accuracy, F1, Precision, Recall, Confusion Matrix
│   ├── soft_gating.py                  # Differentiable soft hierarchical gating
│   └── visualization.py                # Plotting utilities
├── scripts/
│   ├── run_all_baselines.sh            # Run 9 baseline models
│   ├── run_proposed.sh                 # Run AquaNet v3 training
│   └── run_ablation.sh                 # Run 5-variant ablation study
└── paper_v1/                           # IEEE LaTeX manuscript draft & figures
```

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
conda activate vision
pip install -r requirements.txt
```

### 2. Dataset Deduplication & Verification
```bash
python scripts/clean_all_datasets.py
```

### 3. Training Baseline Models
```bash
# Train DenseNet121 baseline
python train.py --model densenet121 --epochs 30 --batch-size 32

# Train all baselines via script
bash scripts/run_all_baselines.sh
```

### 4. Training Proposed AquaNet v3
```bash
python train.py --model aquanet_v3 --epochs 50 --batch-size 32 --lr 0.001
```

### 5. Evaluating Models across Datasets D1, D2, D3
```bash
python test.py --checkpoint checkpoints/aquanet_v3_best.pth --dataset-split test
```

### 6. Single Image Inference
```bash
python inference.py --image path/to/water_sample.jpg --checkpoint checkpoints/aquanet_v3_best.pth
```

---

## Phase 3 reproducibility study (historical)

The Phase 3 manuscript is superseded by Phase 4. It is retained because its shared-hyperparameter protocol is primary evidence for the ranking reversal. Machine-readable results are in `phase3_results/`; the historical pipeline is `experiments/phase3_pipeline.py`.

The principal Phase 3 finding is that AquaNet v3 (85.95% mean accuracy) is statistically indistinguishable from ResNet50 (86.10%) and MobileNetV2 (86.03%) under the matched protocol. See `PHASE3.md` for scope and limitations. Earlier `paper_v1` and `paper_v2` directories are retained as historical drafts and should not be used as the current evidence record.

## Research record

The completed Q4 programme and evidence rules are documented in [`AQUANET_Q4_PLAN.md`](AQUANET_Q4_PLAN.md). `RESEARCH_PLAN.md` records the earlier, superseded Q1 plan.
