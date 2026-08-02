# Project Progress & Technical Audit Report: AquaNet Water Quality Classification

**Project Path**: `/workspace/projects/vision/aquanet` (earlier sections reference the former path `/workspace/notebooks/vasundhra/`)  
**Conda Environment**: `vision` (`/workspace/miniconda3/envs/vision/bin/python`)  
**Hardware Accelerator**: NVIDIA A100-SXM4-40GB GPU  
**Document Updated**: August 2, 2026  

> **Reading order.** This document is chronological and records what was believed at each stage,
> including beliefs later shown to be wrong. **Sections 5 and 6 are superseded and their headline
> numbers are invalid** — they are retained with correction banners because the reasons they were
> wrong became the project's main result. **Section 9 (Phase 4) is the current state of record.**

---

## 1. Executive Summary & Journey

This document records the full history, findings, technical decisions, empirical audits, and current status of the **AquaNet Water Quality Classification (WQC)** research project.

### Major Milestones Accomplished:
1. **System & Environment Recovery**: Re-established working state in conda `vision` environment with PyTorch `2.6.0+cu124` on NVIDIA A100 GPU. Fixed missing dependencies (installed `tabulate` and made script executions non-interactive).
2. **10-Model Baseline Benchmark Completed**: Evaluated 3 Classical ML models (Random Forest, KNN, SVM), 6 Deep Learning baselines (ResNet50, DenseNet121, EfficientNet-B0, MobileNetV2, ViT-Tiny, Swin-Tiny), and AquaNet v2.
3. **5-Variant Ablation Study Completed ([ablation.log](file:///workspace/notebooks/vasundhra/ablation.log))**: Evaluated 5 AquaNet architectural variants (`Full`, `Flat`, `SingleScale`, `NoAttn`, `NoFusion`) over 50 epochs alongside inference timing ($6.53\text{ ms} - 15.47\text{ ms}$) and parameter scaling ($2.23\text{M} - 27.52\text{M}$).
4. **Literature & LaTeX Draft Audit**: Reviewed 25 extracted literature papers ([`corpus/extracted/`](file:///workspace/notebooks/vasundhra/corpus/extracted)) and audited manuscript skeleton ([`paper_v1/main_v1.tex`](file:///workspace/notebooks/vasundhra/paper_v1/main_v1.tex)).
5. **Critical Model Diagnosis**: Identified why AquaNet v2 (85.59%) lost to DenseNet121 (87.68%) and ViT-Tiny (87.13%): AquaNet v2 used a custom stem without dense skip connections and suffered from a hard thresholding error cascade in its dual head.
6. **Empirical Dataset Audit (MD5 & dHash Content Analysis)**: Discovered **1,615 duplicate image clusters** (>25% duplicate content) and **166 cross-split data leakage clusters** in the raw dataset where test images were exact byte-for-byte copies of training images.
7. **3-Dataset Complete Audit & Leakage Removal Pipeline Completed ([clean_all_3_datasets.py](file:///workspace/notebooks/vasundhra/clean_all_3_datasets.py))**:
   - **Dataset 1 (`cleaned_water_dataset`)**: 2,799 unique canonical images (1,956 train / 416 val / 427 test).
   - **Dataset 2 (`cleaned_scrapper_finetune`)**: 213 unique real-world scraped images for real-world fine-tuning.
   - **Dataset 3 (`cleaned_scrapper_unseen_test`)**: 146 unique real-world scraped images for zero-shot OOD testing.
   - **Cross-Dataset Data Leakage**: **0 Clusters** (100% Leakage-Free!).

---

## 2. All 3 Cleaned Datasets Breakdown

```
====================================================================================================
Class        | Dataset 1 (Main Benchmark)   | Dataset 2 (Real Finetune) | Dataset 3 (Unseen OOD Test)
====================================================================================================
clean        | 265                          | 37                        | 26                       
algae        | 993                          | 30                        | 20                       
debris       | 316                          | 24                        | 16                       
foam         | 258                          | 52                        | 35                       
oil          | 154                          | 20                        | 14                       
turbid       | 354                          | 29                        | 20                       
uncertain    | 459                          | 21                        | 15                       
====================================================================================================
TOTAL        | 2,799                        | 213                       | 146                      
====================================================================================================
```

---

## 3. Empirical Color & Feature Audit across 7 Water Quality Classes

Statistical RGB & HSV analysis across 50 sample images per class:

```
====================================================================================================
CLASS       | MEAN RGB (R, G, B)        | STD RGB (R, G, B)         | HSV (Mean H, Mean S, Mean V)
====================================================================================================
clean       | R=119.1, G=154.5, B=170.8 | Std=(63.3, 49.9, 52.1)    | H=85.3, S=102.7, V=178.3 (High Blue)
algae       | R=132.4, G=150.9, B=117.4 | Std=(18.8, 16.4, 24.4)    | H=50.6, S=62.1,  V=151.8 (Green Peak)
debris      | R=109.0, G=110.5, B=104.6 | Std=(46.7, 44.9, 46.3)    | H=58.7, S=57.4,  V=119.4 (Dark Neutral)
foam        | R=138.2, G=134.6, B=128.2 | Std=(48.8, 48.6, 51.7)    | H=47.1, S=35.1,  V=141.2 (Low Saturation)
oil         | R=128.6, G=131.0, B=133.6 | Std=(66.7, 63.0, 64.1)    | H=69.5, S=73.7,  V=148.9 (High Variance)
turbid      | R=129.3, G=119.0, B=103.3 | Std=(35.3, 34.4, 36.3)    | H=31.2, S=74.8,  V=134.9 (Brown Hue)
uncertain   | R=111.7, G=137.0, B=150.9 | Std=(55.7, 50.3, 48.2)    | H=83.0, S=93.0,  V=159.2 (Overlaps Clean)
====================================================================================================
```

### Key Insights:
- **`uncertain` vs `clean` Overlap**: `uncertain` (Mean Hue 83.0, Blue 150.9) has a global color distribution nearly identical to `clean` (Mean Hue 85.3, Blue 170.8). Local spatial attention (CSAB) is necessary to spot localized contamination artifacts amidst bluish water.
- **Oil Sheen Variance**: `oil` exhibits high RGB standard deviation ($\text{Std} = 66.7$) due to rainbow iridescence, requiring multi-scale feature extraction (MSRB).
- **Turbid Sediment**: `turbid` shows a low Mean Hue (**31.2**), reflecting brown clay sediment.

---

## 4. Model Redesign Hypothesis: AquaNet v3

### Why AquaNet v2 Lost to DenseNet121:
1. **Lack of Dense Skip Connections**: DenseNet preserves low-level color/texture signals across all layers.
2. **Hard Threshold Error Cascade**: `if P(Contaminated) >= 0.5` discarded Stage 2 predictions whenever Stage 1 had minor confidence errors.

### AquaNet v3 Architecture Enhancements:
1. **DenseNet121 / ConvNeXt Backbone Foundation**: Preserves ImageNet visual priors and dense channel reuse.
2. **MSRB + CSAB Feature Neck**: Multi-scale fluid extraction and cross spatial-channel glare suppression.
3. **Soft Probabilistic Hierarchical Gating**:
   $$P(\text{Clean}) = P_{\text{binary}}(\text{Clean})$$
   $$P(\text{Class}_k) = P_{\text{binary}}(\text{Contaminated}) \times P_{\text{type}}(k) \quad \text{for } k \in \{1..6\}$$
4. **Class-Weighted Focal Loss**: Inverse frequency weighting ($\alpha_k = 1/N_k$, $\gamma=2.0$) to support minority classes (`oil`, `foam`, `clean`).

---

## 5. Master 10-Model Benchmark Results on Cleaned Dataset 1 (2,799 Unique Images)

> ⚠️ **SUPERSEDED — DO NOT CITE.** Every number in this section is invalid. Three independent
> reasons, all established later: (1) the baselines here were trained with focal loss and AquaNet
> with weighted cross-entropy (defect D2); (2) checkpoints were selected on validation NLL, not
> validation macro-F1 (D1); (3) all models shared one learning rate, which systematically
> penalises pretrained baselines (D3 — see §9.3). Swin-Tiny is listed at 40.05% accuracy here;
> under a matched tuning budget it reaches **92.83%** and is the best model in the study. The
> "AquaNet v3 achieves SOTA" claim is withdrawn. Retained only as a record of what was believed.

Below is the complete, empirical, un-leaked benchmark evaluation across all 10 classical, deep learning, vision transformer, and proposed model architectures on **Dataset 1 (100% Leakage-Free)**:

```
===================================================================================================================
Rank | Model Architecture                   | Test Acc (%) | Macro F1 | Weighted F1 | Model Category & Strategy
===================================================================================================================
1 🏆 | AquaNet v3 (Proposed)                 | 90.40%       | 0.8726   | 0.9045      | DenseNet + MSRB/CSAB + Soft Dual Head
2    | AquaNet v2                            | 83.37%       | 0.7857   | 0.8349      | Custom Stem + MSRB/CSAB + Hard Dual Head
3    | ResNet50                              | 82.67%       | 0.7897   | 0.8296      | Residual Deep Convolutional Baseline
4    | MobileNetV2                           | 81.50%       | 0.7914   | 0.8197      | Lightweight Inverted Residual Baseline
5    | EfficientNet-B0                       | 75.18%       | 0.6955   | 0.7464      | Compound Scaled CNN Baseline
6    | DenseNet121                           | 74.24%       | 0.7122   | 0.7539      | Dense Connectivity Baseline
7    | AquaNet-VLM                           | 72.83%       | 0.6753   | 0.7473      | CLIP VLM + Text Anchors + Cross-Attn
8    | ViT-Tiny                              | 65.81%       | 0.5796   | 0.6342      | Pure Vision Transformer Baseline
9    | AquaNet v1                            | 61.36%       | 0.5248   | 0.6078      | Custom Conv Stem + MSRB/CSAB + Flat Head
10   | Swin-Tiny                             | 40.05%       | 0.3072   | 0.3885      | Hierarchical Shifted Window Transformer
===================================================================================================================
```

### Key Technical Takeaways:
1. **AquaNet v3 Achieves State-of-the-Art ($90.40\%$)**: Outperforms standard DenseNet121 ($74.24\%$) by **$+16.16\%$ Accuracy** and **$+0.1604$ Macro F1**, proving the synergistic benefit of combining dense feature reuse with MSRB fluid extraction, CSAB glare suppression, and Soft Probabilistic Dual-Head gating.
2. **Superiority over Pure Vision Transformers**: ViT-Tiny ($65.81\%$) and Swin-Tiny ($40.05\%$) struggle due to lack of inductive bias on modest-sized water quality datasets without massive pretraining datasets.

---

## 6. Synthetic-to-Real Domain Transfer & Real Video Fine-Tuning Results

> ⚠️ **SUPERSEDED.** Single-seed results under the defective protocol of §5, and AquaNet-only —
> no baseline was ever transferred to Dataset 3 for comparison, so nothing here supports a claim
> about AquaNet specifically. Replaced by §9.6, which runs D1→D3 zero-shot and the D2 adaptation
> curve for all ten architectures at 5 and 3 seeds respectively. Under that comparison AquaNet
> ranks **9th of 10** on zero-shot transfer.

Below is the verified performance of **AquaNet v3** fine-tuned on real-world scraped video frames (**Dataset 2**) and evaluated on unseen real-world video frames (**Dataset 3**):

```
===================================================================================================================
Dataset Evaluation Stage                 | Test Acc (%) | Macro F1 | Weighted F1 | Empirical Significance
===================================================================================================================
Dataset 1 Clean Benchmark (Synthetic/Lab) | 90.40%       | 0.8726   | 0.9045      | High visual clarity benchmark
Dataset 2 Fine-Tune Validation (Real)     | 90.48%       | 0.9023   | 0.9048      | Real-world scraped video frames
Dataset 3 Zero-Shot OOD Test (Pre-FT)     | 39.04%       | 0.4005   | 0.3860      | Direct synthetic-to-real domain gap
Dataset 3 Unseen Real OOD Test (Post-FT)  | 57.53% 🚀    | 0.5384   | 0.5611      | +18.49% real-world generalization gain!
===================================================================================================================
```

### Key Domain Generalization Insights:
1. **Massive Synthetic-to-Real Transfer Boost**: Fine-tuning AquaNet v3 on Dataset 2 boosted real-world zero-shot accuracy on Dataset 3 from **39.04% to 57.53%** (**+18.49% absolute accuracy jump**).
2. **Real-World Fine-Tuning Accuracy**: AquaNet v3 reached **90.48% accuracy** on real-world scraped video data (Dataset 2), demonstrating robustness under natural lighting, turbidity, and variable reflections.

---

## 7. Next Steps & Execution Roadmap

1. **LaTeX Manuscript Update (`paper_v1/main_v1.tex`)**: Update result tables with un-leaked benchmark numbers and real-world domain adaptation results.
2. **Qualitative Figure Generation**: Generate confusion matrices and sample predictions for the paper artifact.

---

## Phase 3 Status — July 31, 2026

Phase 3 replaces claim-first manuscript editing with an evidence-first pipeline. The implementation and reproducible dataset audit are complete. Exact MD5 duplicates are zero across all 3,158 images; the conservative dHash screen flags 698 cross-split candidate pairs requiring review. Consequently, prior “100% leakage-free” wording is withdrawn.

The required 12-run GPU suite and manuscript gate are documented in `PHASE3.md`. `paper_v3` must be written only after `phase3_results/phase3_summary.json` contains every planned run. GPU execution is currently pending sufficient shared A100 availability; no incomplete run has been reported as evidence.

## Phase 3 Completion — July 31, 2026

The complete 12-run matrix and aggregate statistics are available in `phase3_results/phase3_summary.json`. Mean test results over seeds 7, 21, and 42 are:

- AquaNet v3: 85.95% ± 1.07 accuracy, 0.8233 ± 0.0157 macro-F1.
- ResNet50: 86.10% ± 1.41 accuracy, 0.8277 ± 0.0152 macro-F1.
- MobileNetV2: 86.03% ± 1.18 accuracy, 0.8292 ± 0.0128 macro-F1.

Paired tests do not show a significant AquaNet advantage. Seed-42 controlled substitutions give 85.71% (full), 87.82% (no MSRB), 86.65% (no CSAB), and 88.99% (flat head), so causal component claims are withheld pending repeated ablations. `paper_v3/main_v3.pdf` is the current compiled manuscript.

## Expanded Q1 Roadmap

> ⚠️ **SUPERSEDED.** Q1/Q2/Q3 targets and the IrrigWater-7 dataset release were withdrawn on
> July 31, 2026. The dataset name and release package cannot be defended — D1 resolves through a
> public Kaggle dataset containing AI-generated imagery whose per-image provenance was destroyed
> during cleaning (`AQUANET_Q4_PLAN.md` §4.1). Target is Q4. `RESEARCH_PLAN.md` remains in force
> as the **evidence-standards** document (§18 especially); its scope and ordering sections are
> superseded by `AQUANET_Q4_PLAN.md`.

See `RESEARCH_PLAN.md` for the authoritative model-selection protocol, statistical requirements, repeated ablations, domain and robustness studies, explainability, figures, tables, and manuscript gates.

---

# 9. Phase 4 — Protocol Repair and Final Programme (August 1–2, 2026)

**This section is the current state of record.** The experiment programme is complete: **222 phase-4
training runs** plus five evaluation stages, zero outstanding failures. What remains is writing.

## 9.1 Why Phase 4 existed

Phase 3 established that AquaNet did not clearly beat its baselines. A code audit of
`experiments/phase3_pipeline.py` found six defects, five of which affected the proposed model and
the baselines differently — so no comparison in the phase-3 table was admissible in either
direction. Phase 4 rebuilt the training protocol in `experiments/phase4_pipeline.py`, leaving the
phase-3 pipeline **unmodified and still runnable**, because the broken protocol is now evidence.

| | Defect | Fix |
|---|---|---|
| D1 | Checkpoint selection on validation NLL, not validation macro-F1 | Selection on validation macro-F1 for all models; `SELECTION_METRIC=val_macro_f1` guard |
| D2 | Baselines got focal loss, AquaNet got weighted CE | One loss (`wce`) swept once on validation, applied to all |
| D3 | One learning rate for a pretrained trunk and 3.5 M fresh parameters | Per-model tuning budget (Stage T) — **this became the paper** |
| D4 | Class balance corrected twice (sampler *and* loss weights) | `BALANCE=weights`, chosen once, applied to all |
| D5 | The ablation grid never crossed the head with the blocks | Full head × MSRB × CSAB factorial (Stage B, 69 runs) |
| D6 | n=3 statistics; per-image predictions never saved | 5 seeds on all headline claims; every run dumps `y_true`/`y_pred`/`y_prob` |

## 9.2 Infrastructure built

- `scripts/run_all.sh` — single-entry stage driver, idempotent (a run whose result JSON exists is
  skipped), nohup-safe, slot-limited GPU job queue, per-job logs, failure tracking in `.runstate/`.
  `DRY_RUN=1` prints a stage's job list without touching the GPU.
- `scripts/config.sh` — every knob in one place, environment-overridable.
- `experiments/phase4_pipeline.py` — repaired trainer, one run per invocation.
- `experiments/select_lr.py` — per-model LR selection on validation macro-F1 only.
- `experiments/eval_suite.py` (Stage D), `eval_transfer.py` (E), `explain.py` (F),
  `aggregate_all.py` (gates, per-stage summaries, final statistics).
- `preflight` asserts the parameter-matched MSRB control is genuinely parameter-matched
  (3,410,944 = 3,410,944, identical BatchNorm count) and aborts otherwise — without this, Stage B
  proves nothing.

Runtime dropped from ~50 min/run to ~6 min/run through batch and worker tuning, making the
222-run programme feasible in about 24 hours of wall-clock.

## 9.3 Stage T — the central experiment (60 runs)

Every model received an **identical tuning budget**: 3 learning rates × 2 backbone multipliers,
one seed, selected on validation macro-F1 only. The spread between each model's best and worst
configuration in that identical budget:

| model | best LR | bb mult | val macro-F1 | spread |
|---|---:|---:|---:|---:|
| resnet50 | 1e-3 | 1.00 | 0.9130 | **+0.1901** |
| convnext_tiny | 3e-4 | 0.10 | 0.9257 | +0.1636 |
| swin_tiny | 1e-4 | 1.00 | 0.9224 | +0.1128 |
| densenet121 | 3e-4 | 1.00 | 0.9077 | +0.0935 |
| deit_small | 1e-3 | 0.10 | 0.9298 | +0.0836 |
| efficientnet_b0 | 3e-4 | 1.00 | 0.8963 | +0.0742 |
| mobilenetv2 | 1e-3 | 0.10 | 0.9097 | +0.0653 |
| aquanet flat/off/off | 1e-4 | 1.00 | 0.8968 | +0.0477 |
| aquanet hier_tf/on/off | 1e-3 | 1.00 | 0.9097 | +0.0396 |
| **aquanet flat/on/on** | 1e-3 | 1.00 | 0.9020 | **+0.0324** |

**AquaNet is the least learning-rate-sensitive model in the study; the baselines are the most.**
A shared-hyperparameter protocol fixed at a value suiting the proposed model costs the baselines
up to 0.19 validation macro-F1 and costs the proposed model 0.03. That is not a neutral choice,
and it is the mechanism behind the entire earlier result.

What the repair was worth, seed-matched, shared-LR → tuned, test macro-F1:

`resnet50 +0.0574` · `efficientnet_b0 +0.0356` · `densenet121 +0.0143` · `mobilenetv2 +0.0136` ·
`swin_tiny +0.0118` · **`aquanet +0.0101`** · `convnext_tiny +0.0000` · `deit_small −0.0002`

Every baseline gained; the proposed model gained least among those that moved. ResNet50 went from
worst-in-study to 4th of 10 purely by being allowed its own learning rate.

## 9.4 Stage P — the main table (50 runs)

Ten configurations × 5 seeds, each at its own validation-selected learning rate.

| model | test macro-F1 | sd | test acc | val macro-F1 | params |
|---|---:|---:|---:|---:|---:|
| **swin_tiny** *(validation-selected)* | **0.9053** | 0.0094 | 0.9283 | 0.9164 | 27.52M |
| deit_small | 0.8871 | 0.0077 | 0.9162 | 0.9160 | 21.67M |
| convnext_tiny | 0.8851 | 0.0102 | 0.9162 | 0.9132 | 27.83M |
| resnet50 | 0.8798 | 0.0152 | 0.9087 | 0.9069 | 23.52M |
| densenet121 | 0.8751 | 0.0140 | 0.9063 | 0.9009 | 6.96M |
| aquanet[hier_tf,msrb=on,csab=off] | 0.8716 | 0.0127 | 0.9040 | 0.9020 | 10.50M |
| aquanet[flat,msrb=off,csab=off] | 0.8701 | 0.0069 | 0.9030 | 0.8996 | 7.61M |
| efficientnet_b0 | 0.8668 | 0.0072 | 0.8965 | 0.8925 | 4.02M |
| **aquanet[flat,msrb=on,csab=on]** *(published architecture)* | 0.8661 | 0.0165 | 0.8979 | 0.8998 | 10.53M |
| mobilenetv2 | 0.8425 | 0.0205 | 0.8815 | 0.8849 | 2.23M |

The published architecture ranks **9th of 10**, beating only MobileNetV2 and not significantly
(+0.0236, *p* = 0.055). It does not beat its own backbone: DenseNet121 reaches 0.8751 at 6.96 M
parameters against AquaNet's 0.8661 at 10.53 M.

Paired seed-level tests vs. the validation-selected model, Holm-corrected over nine comparisons —
AquaNet is beaten **5/5 seeds, *dz* = 4.54, *p*_holm = 0.0085**, one of only four comparisons that
survives correction. Honest caveat: Swin-T's margins over DeiT-S, ConvNeXt-T, ResNet50 and
DenseNet121 do *not* survive Holm. Those five form a statistically indistinguishable top group;
the defensible claim is that AquaNet sits **below** that group, not that Swin-T is best.

## 9.5 Stage B — the ablation that had never been run (69 runs)

Full head × MSRB × CSAB factorial. Marginal means over the grid:

| axis | levels |
|---|---|
| head | flat 0.8585 · hier_tf 0.8542 · hier_naive 0.8540 |
| MSRB | **off 0.8604** · matched 0.8537 · on 0.8528 |
| CSAB | **off 0.8586** · on 0.8527 |

Seed-matched triplets (n=18): MSRB vs the **parameter-matched control** +0.0014 (8/18 wins);
MSRB vs no neck at all −0.0053 (7/18). Both null. The definitive test at tuned learning rates
over 5 seeds (Stage P): full neck vs no neck is **−0.0040, 2/5 wins, *p* = 0.61**.

**Contribution C2 is refuted** — with the parameter-matched control, which is what makes the
refutation credible rather than merely a capacity artifact. 2.9 M parameters that buy nothing.

The hierarchical-head hypothesis was also refuted. The 3-point flat-vs-hierarchical gap that
motivated the whole Phase 4 novelty argument was a phase-3 protocol artifact; under the repaired
protocol, seed-matched over 5 seeds, the gap is **+0.0035 macro-F1, 4/5 seeds, *p* = 0.78**
(flat 0.8560, hier_naive 0.8525). Teacher forcing, `uncertain`-exclusion and the λ mixing sweep
were all run anyway and all show no effect.

## 9.6 Stages D / E / F — decision quality, transfer, interpretability

Evaluated on the 50 Stage P finalists only (`EVAL_STAGE=P`).

**Stage D (decision quality).** Swin-T takes five of six columns; DeiT-S takes false-clean rate
and corruption robustness. AquaNet leads none.

| model | AURC ↓ | acc@90 ↑ | falseClean ↓ | AUROC ↑ | ECE(temp) ↓ | corrDegrade ↓ |
|---|---:|---:|---:|---:|---:|---:|
| swin_tiny | **0.0104** | **0.9578** | 0.0176 | **0.9922** | **0.0301** | 0.0387 |
| convnext_tiny | 0.0141 | 0.9526 | 0.0202 | 0.9871 | 0.0322 | 0.0315 |
| deit_small | 0.0141 | 0.9521 | **0.0135** | 0.9884 | 0.0328 | **0.0255** |
| aquanet[hier_tf,on,off] | 0.0175 | 0.9406 | 0.0218 | 0.9703 | 0.0328 | 0.0843 |
| aquanet[flat,on,on] | 0.0178 | 0.9333 | 0.0295 | 0.9769 | 0.0374 | 0.0811 |
| efficientnet_b0 | 0.0206 | 0.9432 | 0.0181 | 0.9834 | 0.0324 | 0.0507 |
| mobilenetv2 | 0.0229 | 0.9307 | 0.0254 | 0.9816 | 0.0387 | 0.0676 |
| densenet121 | 0.0321 | 0.9453 | 0.0218 | 0.9849 | 0.0345 | 0.0485 |
| resnet50 | 0.0326 | 0.9443 | 0.0171 | 0.9846 | 0.0428 | 0.0636 |
| aquanet[flat,off,off] | 0.0376 | 0.9495 | 0.0269 | 0.9899 | 0.0304 | 0.0479 |

The hoped-for "decision-quality axis the baselines structurally cannot compete on" does not exist:
the separate `P(contaminated)` output gives no better abstention signal than a flat softmax's
max-probability. **The neck actively harms robustness** — both full-neck variants degrade most
under corruption (0.0843, 0.0811 vs 0.0255 for DeiT-S); removing it cuts degradation ~40%.

**Stage E (D1 → D3 zero-shot, 146 held-out real images never used before).** AquaNet ranks
**9th of 10** here too.

| model | D3 macro-F1 | sd | | model | D3 macro-F1 | sd |
|---|---:|---:|---|---|---:|---:|
| swin_tiny | **0.5035** | 0.0583 | | densenet121 | 0.4399 | 0.0280 |
| deit_small | 0.4980 | 0.0339 | | efficientnet_b0 | 0.4349 | 0.0331 |
| convnext_tiny | 0.4955 | 0.0267 | | aquanet[hier_tf,on,off] | 0.4094 | 0.0381 |
| aquanet[flat,off,off] | 0.4787 | 0.0228 | | **aquanet[flat,on,on]** | 0.3957 | 0.0107 |
| resnet50 | 0.4484 | 0.0184 | | mobilenetv2 | 0.3726 | 0.0230 |

The neck's cost is *larger* out of distribution than in: removing it gains **+0.0830** macro-F1 on
D3, versus −0.0040 in-distribution. A block introduced to improve generalisation impedes it.

D2 adaptation curve (D3 macro-F1, mean over seeds 7/21/42): 25% → 0.5961 · 50% → 0.6653 ·
75% → 0.6456 · 100% → 0.7069. The 75% point is non-monotonic; with n=3 on 146 images this is
within noise and must be reported with CIs.

**Stage F (interpretability).** Deletion/insertion AUC. The one axis where an AquaNet variant
leads (`hier_tf` gap +0.1305), but this must not be turned into a claim: the transformers' near-zero
gaps reflect a known Grad-CAM limitation on attention architectures, not worse localisation.
CSAB mask figures are cut — Stage B shows CSAB does nothing.

## 9.7 Defects found and fixed *in the Phase 4 analysis code itself*

Recorded because each silently corrupted a report before being caught:

1. **Stage D/E/F evaluated the wrong checkpoints.** All 18 report files were built on shared-LR
   screening checkpoints with **zero** Stage P entries. Added `--stage` plumbing through
   `eval_suite.py`, `eval_transfer.py`, `explain.py` and `EVAL_STAGE` in config. Old reports moved
   to `reports/sharedlr_protocol/` — preserved, not deleted, as the "before" arm.
2. **`abstention` and `binary` ignored `--stage`.** They read saved per-image predictions rather
   than checkpoints, bypassing the filter; both reports carried 222 entries mixing every protocol.
   Fixed in `_from_saved_predictions()`.
3. **The final report pooled across stages.** `final_report()` called `load_runs(P4)` unscoped, so
   swin_tiny's summary showed `n_seeds: 14` with seed 42 counted eight times — its Stage T sweep
   points, several deliberately mistuned, averaged into its headline mean. This depressed
   swin_tiny's reported macro-F1 to 0.8872 (true 5-seed value: 0.9053) and made the report
   "select" deit_small. Fixed; `--report` now accepts `--stage`.
4. **A degenerate run contaminated Stage D aggregates.** The `hier_tf … lam0.0` run collapsed
   (val macro-F1 0.0250) and inflated hier_tf's mean AURC from 0.0285 to 0.1114. Excluded as
   degenerate, and reported as such rather than silently dropped.
5. **Transient `NVML_SUCCESS INTERNAL ASSERT FAILED`** in the CUDA caching allocator hit 5 eval
   jobs under 2-way concurrency (training ran 110/110 clean). Cleared by rerunning serially at
   `GPU_SLOTS=1`. Not a code defect, but it broke an `&&` chain and silently skipped two stages.

## 9.8 Where the project stands

**The architecture contribution is closed in all three directions** — accuracy, decision quality,
and ablation under parameter-matched control. The programme nevertheless produced a stronger and
more transferable result than the one it set out to find: *the comparison protocol determined the
published outcome, and the size of that effect is now measured across ten architectures.*

Paper reframe, in order of strength:

1. **A quantified protocol effect** — identical tuning budget, ten architectures, per-model spread.
   Not specific to water; reusable by anyone comparing a custom model against pretrained baselines.
2. **A negative architecture result with proper controls** — parameter-matched neck, full
   factorial, 5 seeds, Holm correction.
3. **A benchmark for the task** — 10 architectures × 5 seeds across accuracy, calibration,
   abstention, false-clean rate, corruption, transfer and interpretability, with a deployment
   recommendation (Swin-T; DenseNet121 at 6.96 M if compute is constrained).

Estimated Q4 acceptance on this framing: **70–75%**. Time to submission: **6–8 weeks, all writing.**

### Outstanding

- Draft the paper against `AQUANET_Q4_PLAN.md` §2.2. Retitle — the current title names a model
  that does not work.
- **Withdraw `paper_v3`** as a claim-bearing document; its central result does not survive Stage T.
- Optional, ~30 min: the natural-only / generated-only training split — the one planned item the
  programme did not execute (`AQUANET_Q4_PLAN.md` §4.2).
- Optional, ~10 min: `pip install fvcore`, re-run `eval_suite.py --task complexity --stage P` to
  fill the null GFLOPs column. Params and GPU/CPU latency are already recorded.

### Artifacts

```
phase4_results/*.json     222 runs: A=20, B=69, C=23, T=60, P=50 (metrics + full config + env + git sha)
predictions/phase4/*.json per-image y_true / y_pred / y_prob for McNemar and bootstrap
checkpoints/phase4/       per-run weights
reports/stageD_*.json     calibration, abstention, binary, corruptions, complexity (Stage P scope)
reports/stageE_*.json     D3 zero-shot + D2 adaptation curve
reports/stageF_*.json     deletion/insertion + 50 Grad-CAM figures
reports/final/            final_report.json — main table, paired tests, McNemar, bootstrap
reports/sharedlr_protocol/  the superseded shared-LR evaluations, kept as the "before" arm
logs/<job>.log            one per run
```
