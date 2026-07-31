# Phase 3: Evidence-First Manuscript Rebuild

## Objective

Produce a reproducible experimental record before writing `paper_v3`. The new manuscript must be generated only after all required artifacts below exist and pass consistency checks.

## Verified starting evidence

- D1 contains 2,799 images: 1,956 train, 416 validation, and 427 test.
- D2 contains 213 real-world fine-tuning images.
- D3 contains 146 held-out real-world images.
- Existing seed-42 AquaNet v3 result: 90.40% accuracy and 0.8726 macro-F1 on D1 test.
- Existing domain transfer: 39.04% before fine-tuning and 57.53% after D2 fine-tuning on D3.
- The new reproducible MD5 audit finds zero exact duplicate groups across all 3,158 files.
- A 64-bit dHash screen at Hamming distance <=5 flags 698 cross-split candidate pairs. These are candidates, not confirmed leakage, and require source-group/manual review. Phase 3 therefore prohibits the phrase "100% leakage-free."

## Required experiments

1. Three-seed comparison (`7`, `21`, `42`) for AquaNet v3, ResNet50, and MobileNetV2.
2. Controlled seed-42 ablations using the same DenseNet backbone and training protocol:
   - full AquaNet v3;
   - 1x1 projection replacing MSRB;
   - identity replacing CSAB;
   - flat seven-class head replacing hierarchical inference.
3. Accuracy, macro-F1, weighted-F1, NLL, 15-bin ECE, parameter count, and latency for every run.
4. Paired statistical tests only when complete seed-matched runs exist.
5. Exact-hash and perceptual-candidate audit artifacts.
6. D1/D2/D3 domain-transfer results with a clearly separated image-level D2 validation limitation.
7. Verified figures generated directly from machine-readable results.

## Implementation

`experiments/phase3_pipeline.py` performs the audit, training, evaluation, calibration, latency measurement, aggregation, and paired tests. Each run writes a checkpoint under `checkpoints/phase3/` and a JSON record under `phase3_results/`. Completed JSON runs are skipped on restart.

Run:

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
  /workspace/miniconda3/envs/vision/bin/python experiments/phase3_pipeline.py
```

The allocator flag is required on the current managed host because cached allocation triggers an NVML permission assertion.

## Manuscript gate

Do not create `paper_v3` until `phase3_results/phase3_summary.json` exists and contains all 12 planned runs. Unsupported claims from `paper_v2`—five-seed p-values, ECE 0.042, existing ablation values, Grad-CAM, edge deployment, ONNX INT8, and “state of the art”—must not be copied forward.

## Current execution status

All 12 planned runs are complete and aggregated in `phase3_results/phase3_summary.json`. The evidence-aligned manuscript is compiled at `paper_v3/main_v3.pdf`. The matched results do not demonstrate an AquaNet v3 advantage; see the manuscript and project progress record for the verified interpretation.
