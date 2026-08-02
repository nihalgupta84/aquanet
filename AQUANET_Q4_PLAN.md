# AquaNet: Q4 Execution Plan — Completed Programme

Original: July 31, 2026 · **Rewritten against final data: August 2, 2026**
Scope: `/workspace/projects/vision/aquanet` **only**. No other project is referenced or combined.
Constraints in force: no new data of any kind; no field collection (future work); Q4 journal target.
Relationship to other documents: this supersedes `RESEARCH_PLAN.md` §19 (execution order) and narrows §7/§11/§12 to a Q4 budget. `RESEARCH_PLAN.md` §18 (evidence rules) stays in force verbatim. `PUBLICATION_STRATEGY.md` §3/§4/§6/§7 are withdrawn.

**Status: the experiment programme is complete.** 222 phase-4 training runs plus five evaluation stages, zero outstanding failures. What remains is writing.

> **This document originally opened with the claim "two AquaNet configurations already beat both baselines." That claim is refuted.** The superseded text is preserved in §0.1 rather than deleted, because the reason it was wrong is now the paper's central finding. A plan that had been quietly edited to match its outcome would be worth less than one that records what it predicted.

---

## 0. Outcome

### 0.1 What this document predicted, and what happened

| Prediction (July 31) | Outcome (August 2) | Verdict |
|---|---|---|
| §0: AquaNet-flat and AquaNet-no-MSRB beat ResNet50 and MobileNetV2 on every seed | Under a matched per-model tuning budget, ResNet50 reaches 0.8798 test macro-F1 and beats every AquaNet configuration | **Refuted** |
| §0: *p* = 0.13–0.16 is an n=3 artifact; 5 seeds should reach significance | 5 seeds reached significance — **against** AquaNet (Swin-T beats it 5/5, Holm *p* = 0.0085) | **Refuted, direction reversed** |
| §1-D1: checkpoint selection on validation NLL costs AquaNet 13× more than ResNet50 | Confirmed and fixed. Selection is now validation macro-F1 for every model | **Confirmed** |
| §1-D2: baselines got focal loss, AquaNet got weighted CE | Confirmed and fixed. One loss for all models | **Confirmed** |
| §1-D3: one LR for a pretrained trunk plus 3.5 M fresh parameters penalises whoever has both | Confirmed, and larger than expected — **this became the paper** | **Confirmed, promoted** |
| §1-D5: MSRB/CSAB were never tested under a working head | Confirmed and fixed. Full head × MSRB × CSAB factorial run at 69 runs | **Confirmed** |
| §2: the hierarchical head underperforms flat by 3 points from gradient starvation | The gap is +0.0035 macro-F1, 4/5 seeds, *p* = 0.78 under a repaired protocol. There was no 3-point gap to explain | **Refuted** |
| §2: a corrected head buys a decision-quality axis baselines cannot compete on | Swin-T and DeiT-S beat both AquaNet heads on AURC, false-clean rate, AUROC and ECE | **Refuted** |
| §2: MSRB/CSAB will either help under a parameter-matched control or give a clean negative | Clean negative. Both are null-to-negative | **Confirmed (negative branch)** |
| §5: 80–85% chance of Q4 accept on a model-wins framing | The model-wins framing is unavailable. A different, better-evidenced paper is available | **Refuted** |

### 0.2 The finding

**The published AquaNet result was an artifact of the comparison protocol, and the size of that artifact is measurable.**

Stage T gave all ten models an identical tuning budget — 3 learning rates × 2 backbone multipliers, one seed, selected on validation macro-F1 only. The spread between each model's best and worst configuration in that identical budget:

| model | best LR | bb mult | val mF1 | spread |
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

**AquaNet is the least learning-rate-sensitive model in the study; the baselines are the most.** A shared-hyperparameter protocol is therefore not neutral. Fixed at a value that suits the proposed model, it costs the baselines up to 19 points of validation macro-F1 and costs the proposed model 3.

The mechanism is the one D3 identified, running in the direction D3 did not anticipate. AquaNet carries ~3.5 M freshly-initialised parameters that need a high learning rate; ResNet50 is ~99.9% pretrained and needs a low one applied to the trunk but not throttled to 3e-5. The shared setting `lr=3e-4, mult=0.1` was a reasonable compromise for a model with both kinds of parameters and a poor one for a model with only pretrained ones.

What that repair was worth, seed-matched, shared-LR → tuned, test macro-F1:

`resnet50 +0.0574` · `efficientnet_b0 +0.0356` · `densenet121 +0.0143` · `mobilenetv2 +0.0136` · `swin_tiny +0.0118` · **`aquanet +0.0101`** · `convnext_tiny +0.0000` · `deit_small −0.0002`

Every baseline gained. The proposed model gained least among those that moved at all. The ordering is the finding.

### 0.3 The final comparison

Stage P: ten configurations, five seeds each, every model at its own validation-selected learning rate.

| model | test mF1 | sd | test acc | val mF1 | params |
|---|---:|---:|---:|---:|---:|
| **swin_tiny** *(selected)* | **0.9053** | 0.0094 | 0.9283 | 0.9164 | 27.52M |
| deit_small | 0.8871 | 0.0077 | 0.9162 | 0.9160 | 21.67M |
| convnext_tiny | 0.8851 | 0.0102 | 0.9162 | 0.9132 | 27.83M |
| resnet50 | 0.8798 | 0.0152 | 0.9087 | 0.9069 | 23.52M |
| densenet121 | 0.8751 | 0.0140 | 0.9063 | 0.9009 | 6.96M |
| aquanet[hier_tf,msrb=on,csab=off] | 0.8716 | 0.0127 | 0.9040 | 0.9020 | 10.50M |
| aquanet[flat,msrb=off,csab=off] | 0.8701 | 0.0069 | 0.9030 | 0.8996 | 7.61M |
| efficientnet_b0 | 0.8668 | 0.0072 | 0.8965 | 0.8925 | 4.02M |
| **aquanet[flat,msrb=on,csab=on]** *(published)* | 0.8661 | 0.0165 | 0.8979 | 0.8998 | 10.53M |
| mobilenetv2 | 0.8425 | 0.0205 | 0.8815 | 0.8849 | 2.23M |

The published architecture ranks **9th of 10**. It beats only MobileNetV2, and not significantly (+0.0236, *p* = 0.055).

Paired seed-level tests against the validation-selected model, Holm-corrected across all nine comparisons:

| swin_tiny vs | Δ macro-F1 | wins | *dz* | *p* | *p*_holm |
|---|---:|---:|---:|---:|---:|
| mobilenetv2 | +0.0628 | 5/5 | 4.04 | 0.0008 | **0.0108** |
| **aquanet[flat,msrb=on,csab=on]** | **+0.0392** | **5/5** | 4.54 | 0.0005 | **0.0085** |
| efficientnet_b0 | +0.0385 | 5/5 | 5.16 | 0.0003 | **0.0055** |
| aquanet[flat,msrb=off,csab=off] | +0.0352 | 5/5 | 3.29 | 0.0018 | **0.0199** |
| aquanet[hier_tf,msrb=on,csab=off] | +0.0337 | 5/5 | 1.69 | 0.0194 | 0.1748 |
| densenet121 | +0.0302 | 5/5 | 1.65 | 0.0209 | 0.1748 |
| resnet50 | +0.0255 | 4/5 | 1.15 | 0.0620 | 0.3026 |
| convnext_tiny | +0.0202 | 5/5 | 1.22 | 0.0521 | 0.3026 |
| deit_small | +0.0182 | 5/5 | 1.31 | 0.0433 | 0.3026 |

Read this carefully, because the honest reading is narrower than the ranking suggests. Swin-T's margins over DeiT-S, ConvNeXt-T, ResNet50 and DenseNet121 **do not** survive Holm correction — those five models form a statistically indistinguishable top group. Swin-T's margins over both AquaNet variants and the two small baselines **do** survive. The claim is *"AquaNet sits below the top group,"* not *"Swin-T is the best model."*

**AquaNet does not beat its own backbone.** DenseNet121 at 6.96 M parameters reaches 0.8751; AquaNet at 10.53 M reaches 0.8661. The neck costs 3.5 M parameters and returns nothing.

---

## 1. Diagnosis: six defects — final status

Every one was in `experiments/phase3_pipeline.py`. That file is preserved unmodified and remains runnable: the broken protocol is now primary evidence, not just history.

| | Defect | Status |
|---|---|---|
| **D1** | Checkpoint selection on validation NLL, not validation macro-F1 | **Fixed.** `phase4_pipeline.py` selects on validation macro-F1 for every model; `SELECTION_METRIC=val_macro_f1` guards it in `scripts/config.sh` |
| **D2** | Baselines trained with focal loss, AquaNet with weighted CE | **Fixed.** One loss (`wce`), swept once on validation, applied to all models |
| **D3** | One LR for a pretrained trunk and 3.5 M fresh parameters | **Fixed, and promoted to the paper's central result.** See §0.2 |
| **D4** | Class balance corrected twice (sampler + loss weights) | **Fixed.** `BALANCE=weights`, chosen once, applied to all |
| **D5** | The ablation grid never crossed the head with the blocks | **Fixed.** Full head × MSRB × CSAB factorial, 69 runs (Stage B) |
| **D6** | n=3 statistics, no per-image predictions saved | **Fixed.** 5 seeds on all headline claims; every run dumps `y_true`/`y_pred`/`y_prob`; McNemar and bootstrap available throughout |

D1's original analysis included this line:

> *under the selection rule the plan itself mandates — best validation macro-F1 — AquaNet-full scores 0.8761, the highest of any configuration in the study, ahead of ResNet50's 0.8687.*

That was true of the phase-3 runs and it did not survive contact with a repaired protocol. Under Stage T, ResNet50 reaches validation macro-F1 0.9130 and AquaNet's best variant reaches 0.9097. The defect was real; the advantage it appeared to conceal was not.

---

## 2. The contribution

### 2.1 What was proposed, and why it is withdrawn

The original §2 argued that the project's strongest finding was a mechanistic negative-then-positive result about hierarchical heads:

> *A naive soft-probabilistic hierarchical head underperforms a flat softmax by 3.0 accuracy points on identical features — and we can say precisely why, and fix it.*

Three causes were proposed: (a) gradient starvation of the type head, (b) the `uncertain` class poisoning the binary head, (c) no temperature or annealing.

**The premise is false.** The 3-point gap was a phase-3 protocol artifact. Under the repaired protocol, seed-matched over the 5 Stage A seeds, the flat/hierarchical gap in test macro-F1 is **+0.0035, 4/5 seeds, *p* = 0.78** (flat 0.8560, hier_naive 0.8525) — nothing to explain. All three proposed mechanisms were tested anyway:

- **(a) Teacher forcing.** `hier_tf` vs `hier_naive` across the Stage B factorial: 0.8542 vs 0.8540. No effect.
- **(b) `uncertain` excluded from the binary loss.** Run as a targeted diagnostic. Does not move the head gap, because there is no head gap.
- **(c) λ mixing sweep** over {0, 0.25, 0.5, 0.75, 1.0}. No interior optimum. λ=0 collapses the model entirely (val macro-F1 0.0250) and is excluded from aggregates as degenerate, not reported as a result.

The decision-quality argument is refuted too. On the tuned finalists, Stage D:

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

Swin-T takes five of six columns; DeiT-S takes false-clean rate and corruption robustness. The separate `P(contaminated)` output does not produce a better abstention signal than a flat softmax's max-probability — the premise that it structurally would was never tested before it was written into the plan.

Two narrower statements *are* supported and should be kept:

- **The head choice does not matter.** flat ≈ hier_naive ≈ hier_tf on accuracy, and hier_tf is marginally ahead on AURC (0.0175 vs 0.0178), acc@90 and false-clean rate. Neither head reaches the top group. (An earlier reading that hier was worse on every axis came from stale shared-LR checkpoints and is withdrawn.)
- **The neck actively harms robustness.** Both full-neck AquaNet variants degrade most in the study under corruption — 0.0843 and 0.0811 mean macro-F1 loss versus 0.0255 for DeiT-S, roughly 3×. Removing the neck (`flat/off/off`) cuts degradation to 0.0479, about 40% less than the full architecture. The same pattern holds on real out-of-distribution data: on D3 zero-shot the no-neck variant reaches 0.4787 macro-F1 while the full architecture reaches 0.3957 (§3, Stage E).

### 2.2 What the paper should claim

> Comparisons between a proposed architecture and pretrained baselines are routinely run at a single shared learning rate. We show this is not a neutral choice. A custom architecture carrying a large fraction of freshly-initialised parameters is markedly less sensitive to learning rate than a near-fully-pretrained baseline; a shared setting therefore transfers a systematic advantage to the proposed model. On a seven-class irrigation-water condition benchmark we quantify the effect across ten architectures under an identical tuning budget: the shared-hyperparameter protocol costs the baselines up to 0.19 validation macro-F1 and the proposed model 0.03. Under a matched per-model budget the proposed architecture falls from first to ninth of ten, is beaten by its own unmodified backbone, and its two novel modules are shown by parameter-matched controls to contribute nothing. We release the full 222-run grid, both protocols, and per-image predictions.

Three contributions, in order of strength:

1. **A quantified protocol effect.** Identical tuning budget, ten architectures, spread reported per model. This is the transferable result — it is not about water at all, and it is reusable by anyone comparing a custom model against pretrained baselines.
2. **A negative architecture result with proper controls.** Parameter-matched neck control (`MatchedNeck`, 3,410,944 params, BN count identical to MSRB — asserted in `preflight`), full head × MSRB × CSAB factorial, 5 seeds, Holm correction. Negative results with this level of control are rare enough to be worth publishing.
3. **A benchmark for the task.** Ten architectures × 5 seeds, with calibration, abstention, false-clean rate, corruption robustness, D3 transfer and interpretability axes, plus a deployment recommendation.

### 2.3 What to do with MSRB and CSAB

The original plan said: *"Decide from data. Do not decide from what the title currently says."* The data decided.

The Stage B factorial (69 runs, shared-LR protocol, marginal means over the grid):

| axis | levels |
|---|---|
| head | flat 0.8585 · hier_tf 0.8542 · hier_naive 0.8540 |
| MSRB | **off 0.8604** · matched 0.8537 · on 0.8528 |
| CSAB | **off 0.8586** · on 0.8527 |

Seed-matched triplets across the grid (n=18): MSRB vs the parameter-matched control **+0.0014, 8/18 wins**; MSRB vs no neck at all **−0.0053, 7/18 wins**. Both null.

The definitive test is Stage P, where both arms run at their own tuned learning rate over 5 seeds: `flat/on/on` vs `flat/off/off` is **−0.0040, 2/5 wins, *p* = 0.61** — the full neck is 2.9 M parameters that buy nothing, and cost robustness (§2.1).

C2 is refuted, with the parameter-matched control the original plan correctly insisted on. Report it as a negative result. Ship the smaller model if a model must ship — or better, ship DenseNet121, which beats both.

---

## 3. Experiment programme — complete

222 phase-4 training runs. All stages complete, zero outstanding failures. Everything used data already in `data/`; no image was collected, scraped or generated.

Runner: `./scripts/run_all.sh <stage>`, idempotent, nohup-safe, per-job logs. See `scripts/README.md`.

| Stage | Runs | Status | Outcome |
|---|---:|:--:|---|
| **0** — unblock | 3 | ✅ | Per-image dumps + file logging added; three missing phase-3 runs completed; `phase3_summary.json` regenerated. **Gate PASS** — `flat` held at seeds 7/21 |
| **A** — protocol repair, core comparison | 20 | ✅ | D1–D4 fixed. Under the shared-LR protocol AquaNet beat ResNet50 and MobileNetV2 — later shown to be the protocol artifact, not a result |
| **B** — head × MSRB × CSAB factorial | 69 | ✅ | The ablation D5 showed had never been run. Both modules null-to-negative; head choice immaterial. §2.3 |
| **C** — baseline breadth | 23 | ✅ | +ConvNeXt-T, Swin-T, DeiT-S, EfficientNet-B0, DenseNet121. All three modern baselines beat all 23 AquaNet configurations |
| **T** — per-model tuning budget | 60 | ✅ | 3 LRs × 2 backbone multipliers × 10 models at seed 42, validation-selected. **The paper's central experiment.** §0.2 |
| **P** — 5-seed promotion at tuned settings | 50 | ✅ | The main table. §0.3 |
| **D** — decision quality | eval | ✅ | Calibration, abstention, binary/false-clean, corruptions (9 × 5 severities), complexity. §2.1 |
| **E** — generalisation on D2/D3 | 13 | ✅ | D1→D3 zero-shot + adaptation curve. Below |
| **F** — interpretability | 2 | ✅ | Grad-CAM + quantitative deletion/insertion. Below |
| **report** | — | ✅ | `reports/final/final_report.json`, scoped to Stage P |

Stages D/E/F evaluate the 50 Stage P finalists only (`EVAL_STAGE=P`). Evaluating the screening stages as well would score ~170 checkpoints trained under the protocol Stage T exists to replace.

### Stage E — D1 → D3 zero-shot transfer (146 images, never used before)

| model | D3 macro-F1 | sd | D3 acc |
|---|---:|---:|---:|
| swin_tiny | **0.5035** | 0.0583 | 0.4890 |
| deit_small | 0.4980 | 0.0339 | 0.4699 |
| convnext_tiny | 0.4955 | 0.0267 | 0.4740 |
| aquanet[flat,off,off] | 0.4787 | 0.0228 | 0.4712 |
| resnet50 | 0.4484 | 0.0184 | 0.4288 |
| densenet121 | 0.4399 | 0.0280 | 0.4288 |
| efficientnet_b0 | 0.4349 | 0.0331 | 0.4233 |
| aquanet[hier_tf,on,off] | 0.4094 | 0.0381 | 0.3877 |
| **aquanet[flat,on,on]** | 0.3957 | 0.0107 | 0.3836 |
| mobilenetv2 | 0.3726 | 0.0230 | 0.3603 |

The published architecture is 9th of 10 here as well. Note the neck's cost is *larger* out of distribution than in: removing it moves AquaNet from 0.3957 to 0.4787, +0.0830 macro-F1 — versus −0.0040 in-distribution. A block introduced to improve generalisation measurably impedes it.

D2 adaptation curve (mean over seeds 7/21/42, D3 macro-F1): 25% → 0.5961 · 50% → 0.6653 · 75% → 0.6456 · 100% → **0.7069**. The 75% point is non-monotonic; with n=3 on 146 test images this is within noise and must be reported with CIs, not smoothed.

**Caveat, unchanged and mandatory:** D3 is 146 images across 7 classes (26 clean, 14 oil, 15 uncertain …). Per-class numbers on 14 images are not stable. Report as a generalisation indication with wide bootstrap CIs, never as a benchmark.

### Stage F — interpretability

Deletion/insertion, mean over 5 seeds. Lower deletion AUC and higher insertion AUC indicate saliency that localises evidence the model actually uses.

| model | deletion AUC ↓ | insertion AUC ↑ | gap |
|---|---:|---:|---:|
| aquanet[hier_tf,on,off] | 0.7739 | 0.9044 | **+0.1305** |
| resnet50 | 0.7747 | 0.8914 | +0.1167 |
| aquanet[flat,off,off] | 0.7934 | 0.9004 | +0.1071 |
| efficientnet_b0 | 0.7845 | 0.8824 | +0.0979 |
| densenet121 | 0.8088 | 0.8973 | +0.0886 |
| mobilenetv2 | 0.7857 | 0.8695 | +0.0837 |
| aquanet[flat,on,on] | 0.8304 | 0.8844 | +0.0540 |
| convnext_tiny | 0.8839 | 0.9075 | +0.0236 |
| swin_tiny | 0.8939 | 0.9138 | +0.0198 |
| deit_small | 0.9042 | 0.9049 | +0.0007 |

This is the **one axis where an AquaNet variant leads**, and it must be reported carefully. The transformers' near-zero gaps reflect a known property of Grad-CAM on attention architectures — patch-token gradients give diffuse maps — not necessarily worse localisation. Presenting this as "AquaNet is more interpretable than Swin-T" would be exactly the unsupported saliency claim `RESEARCH_PLAN.md` §14/§18 forbid. Report the CNN group among itself, note the method caveat, and do not build a contribution on it.

CSAB spatial-mask figures are **cut**: Stage B shows CSAB does nothing, and the original plan said not to illustrate a component that does nothing.

---

## 4. Claim boundaries

All three remain in force. §4.1 and §4.2 are unchanged in substance; §4.3 is now partly answered by data.

### 4.1 Do not introduce D1 as a novel dataset

D1 resolves through a symlink to a public Kaggle dataset whose composition includes AI-generated imagery. Original filenames were replaced during cleaning, so per-image provenance is not recoverable. You cannot name, license and release a dataset assembled from sources whose licenses you do not hold and cannot enumerate.

Describe D1 as *"a curated seven-class benchmark assembled from publicly available image sources, including a proportion of synthetically generated imagery"*, cite the Kaggle source, and release **manifests, splits, preprocessing code, evaluation code and the 222-run result grid** — not repackaged images. C1 is withdrawn; curation moves to methodology.

**Title must change.** The original replacement, *"AquaNet: A Hierarchical Attention Network for Visual Irrigation-Water Condition Recognition"*, is a model-first title for a model that does not work. Candidates matching the actual result:

- *"Shared Hyperparameters Are Not a Fair Comparison: Quantifying Protocol Bias in Custom-Architecture Evaluation"*
- *"How a Custom Architecture Wins: A Controlled Study of Tuning-Budget Bias on an Irrigation-Water Benchmark"*

### 4.2 Disclose the synthetic-imagery proportion

Unchanged and still required. A meaningful share of D1 is AI-generated. State the composition in Section 3 and include in Limitations:

> A proportion of D1 consists of synthetically generated imagery. Absolute performance on this benchmark should not be read as an estimate of field performance.

The natural-only / generated-only split was **not run** — it is the one item from the original plan that the programme did not execute. It is cheap (2 training runs) and should be run before submission, or the limitation stated without the ablation and the omission acknowledged.

### 4.3 One honest limitation paragraph on acquisition artefacts

`AUDIT.md` F1 records that a classifier over file metadata alone reaches 68.85% accuracy on D1 test against a 35.13% majority baseline. Every model sees identical inputs and splits, so relative ranking — which is all the paper now claims — is unaffected. Absolute accuracy is inflated relative to field deployment.

Stage D's corruption study and Stage E's D3 transfer are the constructive answers, and both now have numbers: mean corruption degradation 0.0255–0.0843 across models, and D3 zero-shot at 0.37–0.50 macro-F1 against 0.84–0.91 in-distribution. That gap is the honest measure of how much of D1 performance is benchmark-specific.

---

## 5. Probability

Revised against final data. The July 31 estimate of 80–85% for a Q4 accept assumed the model-wins framing, which is unavailable.

| Outcome | P | Notes |
|---|---:|---|
| **Q4 accept**, protocol-finding framing | **70–75%** | Evidence is strong and the finding is genuinely useful; risk is editorial appetite for a negative result, not evidence quality |
| Q4 accept, model-wins framing | **<5%** | Would require suppressing Stage T. Not available |
| Low-Q3 accept after revision | 35–45% | The protocol finding is the kind of thing a methods-minded reviewer values; needs tight writing |
| Q2 (Applied Sciences, Water, Sensors) | 15–20% | Lower than the July estimate — the corruption/abstention axes no longer carry a positive story |
| Reproducibility/benchmark venue | 45–55% | Possibly the better fit; see §6 |

The evidence base is stronger than it was in July — 222 runs, 5 seeds, Holm correction, parameter-matched controls, both protocols preserved and runnable. What changed is that it supports a different paper.

Time to submission: **6–8 weeks**, all writing. No further compute is required beyond the two optional runs in §4.2 and the fvcore FLOP counts.

---

## 6. Venue shortlist

Verify current quartile at submission time.

**Primary (Q4 / low Q3, Scopus-indexed):**
- *IEEE Access* — explicitly publishes negative and reproducibility results; the best fit for §2.2
- *Traitement du Signal* (IIETA), *Ingénierie des Systèmes d'Information* (IIETA)
- *IJACSA*, *IJEECS*
- *Journal of Electronic Imaging* (SPIE) — higher bar, better reputation

**Worth one attempt given the protocol framing:**
- *ReScience C* or an MLRC-style reproducibility track — a natural home for a controlled protocol-bias study
- *Applied Sciences* (MDPI)

**Do not target** IEEE TIM — requires uncertainty quantification against physicochemical reference measurements, which needs data ruled out of scope.

---

## 7. What is explicitly out of scope

Unchanged:

- Any new data collection, scraping, download or generation.
- Field validation and canal-camera deployment — **future work section only**.
- Classical-ML and VLM benchmark families.
- The IrrigWater-7 dataset release package.
- Q1/Q2/Q3 as the primary target.
- Any comparison, transfer or joint claim involving another project.

---

## 8. Relationship to `RESEARCH_PLAN.md`

**`RESEARCH_PLAN.md` is the standards document. This is the execution document.** Where they conflict on **scope or order**, this document wins. Where they touch **evidence standards**, `RESEARCH_PLAN.md` wins — always, including over this document. §18 was binding on every stage and was never relaxed.

| `RESEARCH_PLAN.md` | Status | Where it lives now |
|---|---|---|
| §18 Evidence rules | **In force verbatim** | Held throughout. No test-driven selection at any stage |
| §8 Final model selection | **Satisfied** | Validation macro-F1 only, guarded by `SELECTION_METRIC` |
| §9 Statistical protocol | **Satisfied** | 5 seeds, per-image dumps, McNemar, bootstrap, paired *t*, Holm |
| §10 Metrics | **Satisfied, extended** | Stage D added false-clean rate, risk–coverage, AURC |
| §6 Dataset QC | In force, minus the group-disjoint re-split | Superseded by `AUDIT.md` correction 1 |
| §7 Benchmark families | **Narrowed, satisfied** | Stage C + T + P. Classical ML, VLM, ResNeXt50, MaxViT cut |
| §11 Ablation program | **Satisfied** | Stage B, 69 runs, with parameter-matched control |
| §12 Dataset/domain experiments | **Partly satisfied** | Stage E complete. The natural/generated split (§4.2) was **not run** |
| §13 Robustness | **Satisfied** | Stage D corruptions, 9 types × 5 severities |
| §14 Explainability | **Satisfied, tightened** | Stage F with deletion/insertion; CSAB figures cut per §3 |
| §15 / §16 Figures and tables | Pruned | Only what the stages produced |
| §17 Manuscript structure | **Rescoped** | Compress to ~11 sections for Q4 |
| §19 Execution order | **Superseded and complete** | §3 |
| §2 Dataset identity ("IrrigWater-7") | **Withdrawn** | §4.1 |
| §3 Candidate titles | **Replaced** | §4.1 — model-first titles no longer apply either |
| §4 C1 Named dataset | **Withdrawn** | §4.1 |
| §4 C2 Validated model | **REFUTED** | Stage B + Stage P. Parameter-matched control shows MSRB null (+0.0014, 8/18) and the full neck negative (−0.0040, 2/5, *p*=0.61). Was "in force, untested" in the July version |
| §4 C3 Cross-family benchmark | **Satisfied**, narrowed to CNN + transformer | Stages C/T/P |
| §4 C4 Domain generalisation | **Satisfied** | Stage E |
| §4 C5 Model understanding | **Satisfied** | Stages B, D, F |
| §5 Dataset release package | **Withdrawn** | §4.1. Release manifests, splits, code and the run grid only |

---

## 9. Next actions

All compute is done. What remains is writing, plus two optional cheap runs.

1. **Draft the paper against §2.2.** Protocol finding first, negative architecture result second, benchmark third. The Stage T spread table is Figure 1.
2. **Rewrite the title and abstract** (§4.1). The current title describes a model that does not work.
3. **Optional, ~30 min:** the natural-only / generated-only split (§4.2) — the one planned item not executed.
4. **Optional, ~10 min:** `pip install fvcore` and re-run `eval_suite.py --task complexity --stage P` to fill the null GFLOPs column.
5. **Withdraw `paper_v3`** as a claim-bearing document. Its central result does not survive Stage T.

What must **not** happen: reporting Stage A's shared-LR comparison as the main result, or omitting Stage T. Stage A is in the paper as the "before" arm of the protocol comparison — that is its only valid use, and it is a genuinely valuable one.
