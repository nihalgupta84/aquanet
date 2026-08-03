# MANUSCRIPT_SPEC.md — extend and verify `paper_final/`

**Status: `paper_final/` already exists and compiles.** This is no longer a build-from-
scratch spec. Your job is verification, one major content addition, and a format decision.

**Do not rewrite `main.tex` from scratch.** Its thesis is correct and evidence-backed.

**Sources, priority order:**
1. `reports/stageD_calibration.json` — calibration + fitted temperature
2. `reports/final/final_report.json` — Stage P 5-seed summary + Holm tests
3. `reports/stageD_*.json`, `stageE_*.json`, `stageF_*.json`
4. `predictions/phase4/*.json` — per-image labels/probs/preds
5. `phase4_results/*.json` — per-run metrics, config, env
6. `RESEARCH_PLAN.md` §15/§16 (figure/table lists), §18 (evidence rules — **binding**)

---

## 0. WHAT ALREADY EXISTS

| Artifact | State |
|---|---|
| `paper_final/main.tex` | Complete 8-section draft, IEEEtran, compiles |
| `paper_final/references.bib` | 16 entries, rebuilt clean — fabricated ones removed |
| `paper_final/main.pdf` | Builds |
| `experiments/generate_final_paper_figures.py` | Runs, but **numbers are hardcoded** (GATE F) |
| Title | *When Shared Hyperparameters Reverse Model Rankings* |
| Thesis | Protocol sensitivity as the finding, not AquaNet superiority — **correct, keep it** |

---

## 1. HARD GATES

### GATE A — references (mostly cleared, two checks left)
The fabricated `paper_v1` entries are gone. Remaining work:
- **Verify two DOIs by opening them:** `al2024drone` (Applied Sciences 14(6):2382,
  `10.3390/app14062382`) and `kim2023cctv` (Water 15(18):3308, `10.3390/w15183308`).
  These are the only two domain citations and both carry the paper's gap claim.
- **Remove two uncited entries:** `dosovitskiy2021vit` and `lin2017focal` are in the bib
  but never `\cite`d. Cite them or delete them.
- `RESEARCH_PLAN.md` §18 forbids invented citations. Never re-import anything from
  `paper_v1/references.bib`.

### GATE B — ECE significance test still missing
`aggregate_all.py::final_report()` loops `for metric in ('accuracy', 'macro_f1')`. Add
`'ece_15bin'`, then:
```
python experiments/aggregate_all.py --report --stage P --out reports/final
```
**Sign convention:** lower ECE is better, so the selected model (Swin) shows a *positive*
`mean_diff` meaning AquaNet is BETTER. Inverting this inverts the claim.

Required before §2 below can be stated as significance rather than description.

### GATE C — two ECE sources disagree
`final_report.json` `ece_mean` (from `phase4_pipeline.ece()`) vs
`stageD_calibration.json` `ece_raw` (from `eval_suite.ece_bins()`) differ in the 3rd–4th
decimal — DenseNet121: 0.0603 vs 0.0605. Different pipelines.

**Use `stageD_calibration.json` exclusively.** It is the only source carrying fitted
temperature. Note the agreement-to-3-decimals once in the protocol section.

### GATE F — figures are hardcoded (NEW, most urgent)
`generate_final_paper_figures.py` reads `final_report.json` for the ranking figure only.
Every other number is a literal Python dict:

```python
sense = {"ResNet-50":.1901, ...}            # should come from select_lr.py --table
ood   = {"Swin-T":.5035, ...}               # should come from stageE_zeroshot.json
y     = [.5961,.6653,.6456,.7069]           # should come from stageE_adapt_*.json
aurc  = [.0104,.0141,...]                   # should come from stageD_abstention.json
corrupt = [.0387,.0255,...]                 # should come from stageD_corruptions.json
exp   = {"AquaNet-hier":(.7739,.9044),...}  # should come from stageF_deletion.json
plt.axvline(.866061, ...)                   # magic constant = AquaNet-full mF1
```

**Rewrite every one to load from JSON.** Then regenerate and diff the PDFs against the
committed ones. Any mismatch is a stale number that would have shipped.

`select_lr.py --table` already computes the sensitivity spread — call it or replicate its
logic rather than retyping.

---

## 2. THE MISSING RESULT — highest-value addition

The draft's §5.4 reports only temperature-**scaled** ECE (Swin .0301, AquaNet-full .0374)
and concludes AquaNet is worse. That uses the wrong AquaNet variant and the wrong ECE.

From `stageD_calibration.json`, sorted by raw ECE:

| Model | Fitted T | ECE raw | ECE calibrated |
|---|---:|---:|---:|
| **AquaNet-no-neck** | **1.098** | **0.0397** | 0.0304 |
| Swin-T | 2.300 | 0.0528 | 0.0301 |
| DenseNet-121 | 1.591 | 0.0605 | 0.0345 |
| ConvNeXt-T | 1.987 | 0.0608 | 0.0322 |
| DeiT-S | 1.863 | 0.0621 | 0.0328 |
| ResNet-50 | 1.836 | 0.0642 | 0.0428 |
| AquaNet-hier | 1.619 | 0.0692 | 0.0328 |
| AquaNet-full | 1.640 | 0.0700 | 0.0374 |
| EfficientNet-B0 | 2.753 | 0.0763 | 0.0324 |
| MobileNetV2 | 2.390 | 0.0879 | 0.0387 |

**AquaNet-no-neck is first of ten on uncalibrated ECE and needs a near-ideal T = 1.098
against 1.591–2.753 for every baseline.** The gap to second place (1.591) is the largest
single margin in the study.

Why this **strengthens** the existing thesis rather than competing with it:
- It is another axis on which **removing the neck wins**: T goes 1.640 → 1.098 and raw ECE
  0.0700 → 0.0397 when MSRB and CSAB come off. §5.3's "modules are null-to-negative"
  becomes "null-to-negative on accuracy and actively harmful to calibration."
- It gives the paper one honest positive result, which strengthens a negative-results
  paper — it shows the evaluation was capable of detecting a win.
- It is not cherry-picking, because the same table already reports AquaNet 9th of 10.

**Honest limits, both mandatory in the text:**
- After temperature scaling everything converges to 0.0301–0.0428 and Swin edges AquaNet
  by 0.0003. State this explicitly.
- AquaNet-no-neck is less accurate (0.8701 vs 0.9053). Report accuracy in the same table.
- A merely timid model would show T < 1. AquaNet's T = 1.098 is slightly above 1, so it is
  near-ideal, not underconfident. Make this argument — it pre-empts the obvious objection.

**Deployment framing:** a canal-side camera has no held-out validation split in the field,
so temperature cannot be fitted post-deployment. T ≈ 1.1 ships usable as-is; T ≈ 2.3
reports roughly double the confidence it has earned. Do not overstate — no field trial ran.

**New figure required:** fitted-temperature bar chart, full column width, reference line at
T = 1.0.

---

## 3. FORMAT DECISION — ask the author before proceeding

The draft is ~6 pages, 8 sections, letter-style. `RESEARCH_PLAN.md` §17 specifies 19
sections, and the target (IEEE Access / Traitement du Signal / J. Electronic Imaging) is
long-format.

**Recommendation: expand to 19 sections, keep the thesis unchanged.** The argument is
right; the length is wrong for the venue.

| Current | §17 target |
|---|---|
| Introduction | 1. Introduction |
| Related Work | 2. Related Work and Taxonomy |
| Data and Task | 3. Dataset · 4. Problem Formulation |
| Models and Controlled Protocol | 5. AquaNet Evolution · 6. Final Architecture · 7. Experimental Protocol |
| §5.1–5.2 | 9. CNN Benchmark · 10. Transformer Benchmark |
| §5.3 | 11. Extensive Ablations |
| §5.4 | 14. Robustness and Calibration ← **expand heavily, see §2** |
| §5.5 | 13. Domain Generalization |
| §5.6 | 15. Interpretability |
| — | 8. Classical ML Benchmark ← **GATE E** |
| — | 12. Dataset Contribution Analysis |
| — | 16. Complexity and Deployment |
| Discussion | 17. Discussion |
| Limitations and Ethics | 18. Limitations |
| Conclusion | 19. Conclusion |

### GATE E — two target sections have no data
`AQUANET_Q4_PLAN.md` dropped the classical ML and VLM families; Stage P contains neither.
- **§8 Classical ML** — run it or delete it. Running is cheap: LBP + HOG + GLCM + RGB/HSV
  histograms → LogReg / RF / RBF-SVM / XGBoost, CPU-minutes on 2,799 images, and
  `models/classical/` already exists. It also strengthens §12's shortcut-learning argument.
- **VLM** — delete. Do not invent a CLIP row.

---

## 4. VERIFIED NUMBERS

Stage P finalists, 5 seeds (7, 21, 42, 1337, 2024). Selection on **validation** macro-F1.

| Model | Params (M) | Val mF1 | Test Acc ± sd | Test mF1 ± sd |
|---|---:|---:|---:|---:|
| Swin-Tiny | 27.52 | 0.9164 | 0.9283 ± 0.0082 | 0.9053 ± 0.0094 |
| DeiT-Small | 21.67 | 0.9160 | 0.9162 ± 0.0053 | 0.8871 ± 0.0077 |
| ConvNeXt-Tiny | 27.83 | 0.9132 | 0.9162 ± 0.0071 | 0.8851 ± 0.0102 |
| ResNet50 | 23.52 | 0.9069 | 0.9087 ± 0.0125 | 0.8798 ± 0.0152 |
| DenseNet121 | 6.96 | 0.9009 | 0.9063 ± 0.0116 | 0.8751 ± 0.0140 |
| AquaNet-hier | 10.50 | 0.9020 | 0.9040 ± 0.0094 | 0.8716 ± 0.0127 |
| AquaNet-no-neck | 7.61 | 0.8996 | 0.9030 ± 0.0069 | 0.8701 ± 0.0069 |
| EfficientNet-B0 | 4.02 | 0.8925 | 0.8965 ± 0.0061 | 0.8668 ± 0.0072 |
| AquaNet-full | 10.53 | 0.8998 | 0.8979 ± 0.0129 | 0.8661 ± 0.0165 |
| MobileNetV2 | 2.23 | 0.8815 | 0.8815 ± 0.0159 | 0.8425 ± 0.0205 |

**Statistical facts**
- Test set = **427 images** (accuracy differences are exact multiples of 1/427).
- D1: 2,799 total, splits 1,956 / 416 / 427. D2 = 213. D3 = 146 (14–26 per class).
- Under Holm across 9 comparisons, Swin is **NOT** significantly better than DeiT-S,
  ConvNeXt-T, ResNet-50 or DenseNet-121 (p_holm = .303, .303, .303, .175). **Top five are
  statistically indistinguishable.** The draft states this correctly — keep it.
- Swin IS significantly better than AquaNet-full (.0392, 5/5, p_holm = .0085) and
  AquaNet-no-neck (.0352, 5/5, p_holm = .0199).
- Wilcoxon not reportable at n=5 (min attainable two-sided p = 0.062).
- Prediction-level bootstrap + McNemar exist for **seed 7 only**. Label them as such.

**Ablation (§11)** — adding MSRB + CSAB to AquaNet-no-neck:

| Metric | no-neck | full | Direction |
|---|---:|---:|---|
| Test macro-F1 | 0.8701 | 0.8661 | worse |
| ECE raw | 0.0397 | 0.0700 | much worse |
| Fitted T | 1.098 | 1.640 | much worse |
| Seed sd (mF1) | 0.0069 | 0.0165 | 2.4× more variance |
| Corruption degradation | 0.0479 | 0.0811 | worse |
| D3 zero-shot mF1 | 0.4787 | 0.3957 | worse |
| Params | 7.61M | 10.53M | +2.92M |

**Every axis degrades.** This is the paper's cleanest result. `MatchedNeck` in
`phase4_pipeline.py` is the parameter-matched control separating "MSRB helps" from "3.4M
extra parameters help" — describe it explicitly; it is what makes the negative result
publishable.

**Verify from raw JSON before use** (currently hardcoded in the figure script): protocol
sensitivity spreads, D3 zero-shot per model, adaptation curve, AURC, corruption
degradation, deletion/insertion AUC, false-clean rate (**DeiT-S wins this, not AquaNet**).

---

## 5. CLAIM BOUNDARIES (`RESEARCH_PLAN.md` §18)

**Never write:** AquaNet is most accurate / SOTA / outperforms baselines · a novel dataset
is released · MSRB or CSAB improve performance · Swin is definitively best · data is
leakage-free · any physicochemical parameter is measured · deployment-ready or
field-validated · D3 is a benchmark · causality from rankings.

**Safe to write:** AquaNet-no-neck attains the lowest uncalibrated ECE and near-ideal
fitted temperature among ten architectures over five seeds · at 7.61M params, 3.6× fewer
than Swin-T · after temperature scaling models converge and Swin edges AquaNet by 0.0003 ·
AquaNet ranks 9th of 10 on macro-F1 · MSRB/CSAB are null-to-negative against a
parameter-matched control and degrade calibration, seed stability, corruption robustness
and OOD transfer · the top five are statistically indistinguishable under Holm · identical
hyperparameter values are not a fair comparison across architectures with differing
pretrained/fresh parameter ratios · external-domain performance is substantially below
in-distribution.

**Dataset framing (the Q4 banner in `RESEARCH_PLAN.md` retracts C1):** D1 originates from
the public Kaggle dataset `vasundharadixit1826/real-and-ai-data` and contains AI-generated
imagery, so it "cannot be introduced, named or released as a novel dataset." Write "we
consolidate, audit and extend", never "we introduce". The draft already handles this
correctly — do not weaken it. Confirm with the author whether that Kaggle account is
co-author Vasundhra Dixit.

---

## 6. FIGURE AND TABLE INVENTORY (`RESEARCH_PLAN.md` §15/§16)

Existing: `ranking.pdf`, `protocol_sensitivity.pdf`, `ood_transfer.pdf`, `adaptation.pdf`,
`decision_quality.pdf`, `explanations.pdf`, `gradcam_aquanet_seed42.png`. All except
`ranking.pdf` need de-hardcoding (GATE F).

Add for the long format:

| Figure | Source | Status |
|---|---|---|
| **Fitted temperature bar chart, line at T=1.0** | `stageD_calibration.json` | **HEADLINE** |
| Reliability diagrams | `stageD_calibration.json` `reliability_raw` | generate |
| Confusion matrices (Swin + AquaNet-no-neck) | `predictions/phase4/` | generate |
| Per-class F1 grouped bars | `predictions/phase4/` | generate |
| ROC / PR one-vs-rest | `predictions/phase4/` `y_prob` | generate |
| Pareto: accuracy vs params vs latency | `final_report` + `stageD_complexity` | generate |
| Ablation heatmap (head × msrb × csab) | `phase4_results/` Stage B | generate |
| Full corruption curves, 9 × 5 | `stageD_corruptions.json` | generate |
| Class / source / resolution distributions | dataset audit JSON | generate |
| Sample grid, 7 classes | dataset | generate |
| Architecture, MSRB, CSAB, MatchedNeck diagrams | source | draw |

Tables to add: dataset composition, training protocol, per-class performance, **calibration
(headline — T, ECE raw, ECE cal, NLL, Brier, plus accuracy)**, complexity, statistical
comparisons.

**Complexity table:** if `gflops` is null, fvcore was unavailable —
`eval_suite.task_complexity` writes `gflops_note` when it fails. Install fvcore and re-run,
or omit the column. Never estimate FLOPs analytically and present them as measured.

---

## 7. WRITING RULES

- Keep `\documentclass[10pt,journal]{IEEEtran}`.
- Replace `\author{Anonymous Authors}` with real authors and affiliations before submission.
- Every numeric claim traces to a file. Comment the source above each table and figure.
- Report mean ± sd with n=5 and the seed list. Never headline a single seed.
- Never write "significantly" without a Holm-corrected p supporting it.
- Missing evidence → `\todo{...}`. Never fill a gap with plausible text.
- Do not reuse prose from `paper_v1`/`v2`/`v3` — their framing is contradicted.

---

## 8. ORDER OF WORK

1. GATE F — de-hardcode the figure script, regenerate, diff against committed PDFs
2. GATE B — add `ece_15bin` to the paired test, re-run aggregation
3. §2 — add the calibration result to §5.4 and build the temperature figure
4. GATE A — verify the two DOIs, drop the two uncited entries
5. Format decision with the author (short vs 19-section)
6. GATE E — run classical ML or delete §8
7. Expand to long format, generate remaining figures and tables
8. Author metadata, acknowledgements, data-availability statement

**If this spec conflicts with what the JSON files contain, the JSON wins.** Record the
conflict in `paper_final/BUILD_NOTES.md` rather than silently resolving it.
