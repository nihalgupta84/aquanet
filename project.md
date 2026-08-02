# Project Journal — AquaNet Water Quality Classification

> Detailed project log documenting architectural decisions, dataset audits, baseline benchmarks, ablation experiments, and paper development.

---

## 2026-07-30 — Project Initialization & Structure Setup

### Key Decisions Made
1. **Project Name**: `aquanet`
2. **Project Path**: `/workspace/projects/vision/aquanet`
3. **Target Paper**: IEEE Transactions manuscript draft (`paper_v1/main_v1.tex`)
4. **Hardware Environment**: NVIDIA A100-SXM4-40GB GPU in `vision` conda environment (PyTorch 2.6.0+cu124)
5. **Dataset Handling**: Audited all 3 dataset splits using MD5 and dHash. Discovered 1,615 duplicate clusters and 166 cross-split data leakage clusters in the raw dataset. Purged all leakage and created a 100% clean 2,799 unique image dataset (`./data/cleaned_water_dataset`).
6. **Model Redesign (AquaNet v3)**: Diagnosed why AquaNet v2 (85.59%) underperformed DenseNet121 (87.68%). Designed **AquaNet v3** featuring a DenseNet121 backbone, MSRB multi-scale fluid neck, CSAB spatial-channel attention block, and Soft Probabilistic Hierarchical Dual-Head gating.

---

## Architecture Design: AquaNet v3

- **Backbone Foundation**: Pretrained DenseNet121 (preserves low-level RGB/HSV channel features alongside high-level semantics).
- **MSRB Feature Enhancement Neck**: Multi-scale parallel convolutions ($1\times1$, $3\times3$, dilated $3\times3$, AvgPool) to capture multi-scale fluid phenomena (oil sheen, micro-foam, algae).
- **CSAB Attention**: Cross Spatial-Channel Attention to suppress water surface glare and sky reflections.
- **Soft Probabilistic Gating**:
  $$P(\text{Clean}) = P_{\text{head1}}(\text{Clean})$$
  $$P(\text{Class}_k) = P_{\text{head1}}(\text{Contaminated}) \times P_{\text{head2}}(k) \quad \text{for } k \in \{1..6\}$$
  *Eliminates the hard step function error cascade of AquaNet v2.*

---

## Project Structure Overview

```
/workspace/projects/vision/aquanet/
├── project.md                          # Developer journal & log
├── project_progress.md                 # Detailed progress & empirical audits
├── README.md                           # User-facing project documentation
├── requirements.txt                    # Dependencies
├── environment.yml                     # Conda specification
├── train.py                            # Unified training entrypoint
├── test.py                             # Unified evaluation entrypoint
├── inference.py                        # Single-image / batch inference
├── configs/                            # Configuration files
├── data/                               # Clean 3 datasets (D1, D2, D3)
├── dataset/                            # PyTorch Dataset & Transforms
├── models/                             # Classical, DL, Transformer, & AquaNet v3 models
├── utils/                              # Metrics, Seed, Logger, Soft Gating, Visualization
├── scripts/                            # Execution bash & audit scripts
├── paper_v1/                           # IEEE LaTeX manuscript & figures
├── corpus/                             # Extracted literature extractions (25 papers)
├── results/                            # Metrics & evaluation outputs
├── checkpoints/                        # Model weights
└── logs/                               # Execution logs
```

---

## 2026-07-31 — Phase 3 Evidence-First Rebuild

The manuscript audit found that `paper_v2` mixed verified seed-42 results with unsupported ablation, statistical significance, calibration, interpretability, and deployment claims. Phase 3 was established to regenerate evidence before drafting `paper_v3`.

Implemented `experiments/phase3_pipeline.py` with reproducible MD5/dHash auditing, controlled AquaNet v3 component ablations, three-seed baseline comparisons, calibration metrics, latency/parameter reporting, aggregation, and paired tests. The audit confirms zero exact duplicate groups across D1/D2/D3 but identifies 698 perceptual candidate pairs at dHash distance <=5; future writing must not call the data “100% leakage-free.”

The GPU suite is resumable but was paused without partial results due to unrelated shared A100 workloads. See `PHASE3.md` for the manuscript gate and exact experiment requirements.

## 2026-07-31 — Phase 3 Experiments and Manuscript Completed

All 12 planned runs completed: three matched seeds each for AquaNet v3, ResNet50, and MobileNetV2, plus seed-42 no-MSRB, no-CSAB, and flat-head substitutions. AquaNet v3 averaged 85.95% accuracy, compared with 86.10% for ResNet50 and 86.03% for MobileNetV2; paired tests found no significant difference. Single-seed substitutions did not validate a positive contribution from every AquaNet component.

The result-derived figures and evidence-aligned IEEE draft are in `paper_v3/`. The manuscript explicitly separates the earlier 90.40% retained run from the matched Phase 3 protocol, withdraws unsupported claims, and foregrounds the real-domain gap and provenance limitations.

## 2026-07-31 — Q1 Dataset-and-Model Paper Reframing

The authoritative plan is now `RESEARCH_PLAN.md`. The paper is reframed around the provisional IrrigWater-7 dataset, a validation-selected AquaNet configuration, a four-family benchmark, domain generalization, robustness, calibration, and explainability. Leakage auditing and reproducibility remain quality-control practices rather than novelty claims. The current short `paper_v3` is an interim artifact and will be replaced only after the expanded evidence gates are satisfied.
