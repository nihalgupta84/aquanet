# BUILD_NOTES.md

Record of every point where `MANUSCRIPT_SPEC.md`, the previous `main.tex`, or `AUDIT.md`
disagreed with what the result JSONs actually contain. Per `MANUSCRIPT_SPEC.md` §8, **the
JSON wins** and the conflict is recorded here rather than silently resolved.

Build date: 2026-08-03. Every number below is reproducible with:

```bash
python experiments/generate_final_paper_figures.py
python experiments/generate_final_paper_tables.py
```

Both scripts read only `reports/`, `phase4_results/` and `predictions/` through
`experiments/paper_data.py`. Neither contains a result literal.

---

## GATE F — hardcoded figure numbers (resolved, 2 stale values found)

The previous `generate_final_paper_figures.py` carried 47 result literals. Each was checked
against its JSON source at a tolerance of 5e-5:

| Figure | Literals | Stale |
|---|---:|---:|
| `protocol_sensitivity.pdf` | 10 | 0 |
| `ood_transfer.pdf` | 10 | 0 |
| `adaptation.pdf` | 4 | 0 |
| `decision_quality.pdf` | 10 | 0 |
| `explanations.pdf` | 12 | **2** |
| `ranking.pdf` axvline | 1 | 0 |

**The two stale values would have shipped:**

| Quantity | Hardcoded | `stageF_deletion.json` | Error |
|---|---:|---:|---:|
| AquaNet-full deletion AUC | 0.8330 | 0.8304 | +0.0026 |
| AquaNet-full insertion AUC | 0.8870 | 0.8844 | +0.0026 |

`main.tex` never quoted these two numbers in prose, so the defect was confined to
`explanations.pdf`. Both are now loaded from JSON.

## GATE B — ECE significance test (added; result is *not* significant)

`aggregate_all.py::final_report()` now loops `('accuracy', 'macro_f1', 'ece_15bin')`. Two
changes beyond adding the metric:

- Holm is applied **within** each metric family (9 comparisons each), not across all 27.
  Pooling would correct across comparisons that answer different questions.
- Every record carries `lower_is_better` and an explicit `favours` field. `mean_diff` is
  always *selected minus other*, so on a loss a positive delta means the other model won.
  The field removes the chance of reading the sign backwards.

Result, Swin-T (the validation-selected model) against AquaNet-no-neck on `ece_15bin`:

```
delta +0.0131   wins 1/5   p = 0.0623   p_holm = 0.3114   favours: other
```

**`MANUSCRIPT_SPEC.md` §2 asks whether the calibration result can be stated as significance.
It cannot.** The same test on the evaluation-suite ECE gives p = 0.0831, p_holm = 0.3803.
The manuscript therefore states the ECE result as a **ranking over ten architectures**,
which is what the "safe to write" list in §5 of the spec already licenses, and never as a
significant difference.

The **fitted temperature** result *is* significant. Testing |T − 1| pairwise across the five
seeds, AquaNet-no-neck is closer to the ideal temperature than **all nine** other models,
5/5 seeds in eight of nine cases, Holm-corrected p ≤ 0.0276 throughout (worst case
DenseNet-121). This, not the ECE ranking, is what carries the calibration section.

## GATE C — the two ECE sources (resolved)

`final_report.json::ece_mean` (from `phase4_pipeline.ece()`) and
`stageD_calibration.json::ece_raw` (from `eval_suite.ece_bins()`) are independent code paths.

**Correction to the spec:** `MANUSCRIPT_SPEC.md` §C says the two "agree to 3 decimals". They
do not, quite. Maximum absolute disagreement is **4.75e-4** (EfficientNet-B0: 0.07676 vs
0.07629), which rounds to 0.077 against 0.076 at three decimals. MobileNetV2 (4.4e-4) is the
same case.

Accurate statement, used in the manuscript: *the two implementations agree to within
5 × 10⁻⁴ on all ten models.* The manuscript quotes `stageD_calibration.json` exclusively, as
the spec requires, since it is the only source carrying the fitted temperature.
`tables/ece_sources.tex` reports the full comparison.

## GATE A — references (resolved; **both domain citations were partly fabricated**)

Every one of the 16 entries was checked on 2026-08-03 against Crossref (by DOI) or the arXiv
API. The two DOIs the spec asked about both resolve to real articles on the stated topics.
**But both entries carried invented author given names:**

| Key | Was in `references.bib` | Actual (Crossref) |
|---|---|---|
| `al2024drone` | Al-Battbootti, **Khalid** and others | Al-Battbootti, **Myssar Jabbar Hammood**; Marin, Iuliana; Al-Hameed, Sabah; Popa, Ramona-Cristina; Petrescu, Ionel; Boiangiu, Costin-Anton; Goga, Nicolae |
| `kim2023cctv` | Kim, **Young-Jin** and Choi, **Min-Ha** | Kim, **Kwihoon**; Choi, **Jin-Yong** |

`al2024drone`'s title was also truncated — the subtitle *"Case Study in Shatt al-Arab, South
East Iraq"* was missing. Both are corrected, with DOIs, and the file now carries a header
recording the verification date and method. The other 14 entries verified clean; full author
lists and DOIs were added where a DOI exists.

`dosovitskiy2021vit` and `lin2017focal` were uncited. Rather than delete them, both are now
genuinely cited — ViT in §2 as the lineage for DeiT and Swin, focal loss in §7 where the
loss protocol explains why a single weighted cross-entropy was fixed for all models.

## Stage B factorial — **corrected: 54 balanced runs, not 69**

69 files carry `stage == "B"`. They are not 69 factorial replicates:

- **54** form the balanced design: 3 heads × 3 MSRB levels × 2 CSAB levels × 3 seeds, all at
  the base configuration (`lambda_mix = 0.5`, `uncertain_binary = contam`). 18 cells, 3 runs
  each, perfectly balanced.
- **15** are one-off sensitivity runs that exist for the `(hier_tf, on, on)` cell **only**:
  a λ_mix sweep (0.0, 0.25, 0.75, 1.0 × 3 seeds) and an `uncertain_binary = exclude` variant
  (3 seeds).

The previous `main.tex` marginals were computed over an unbalanced 57-run subset
(λ_mix = 0.5, which keeps the three `uexcl` runs). That lets one cell's side experiment move
an axis it does not belong to:

| Marginal | `main.tex` (57 runs) | Balanced (54 runs) |
|---|---:|---:|
| head flat | .8585 | .8585 |
| head hier-naive | .8540 | .8540 |
| head hier-TF | .8542 | **.8534** |
| MSRB off | .8604 | .8604 |
| MSRB matched | .8537 | .8537 |
| MSRB on | .8528 | **.8518** |
| CSAB off | .8586 | .8586 |
| CSAB on | .8527 | **.8520** |

The matched-neck comparison changes more consequentially:

| Comparison | `main.tex` | Balanced design |
|---|---|---|
| MSRB vs MatchedNeck | **+.0014**, 8/18 | **−.0018**, 6/18 |
| MSRB vs no neck | −.0053, 7/18 | −.0086, 6/18 |

**The sign of the MatchedNeck comparison flips.** Both values are null-sized, so the paper's
conclusion is unchanged — but it moves from "MSRB is marginally ahead of its
parameter-matched control" to "MSRB is marginally behind it", which if anything strengthens
the negative result. The manuscript now reports the balanced numbers and describes the 15
auxiliary runs separately, in §11.

One λ_mix point is worth its own sentence and gets one: at λ_mix = 0 the hierarchical
objective is switched off entirely while `forward_probs` still routes inference through the
binary × type gating product, so the two heads are never trained and macro-F1 collapses to
0.0247 ± 0.0007. That is a wiring sanity check, not a result about head design, and it is
labelled as such.

## `MANUSCRIPT_SPEC.md` §4 table — one transcription error

59 of 60 values in the spec's Stage-P table match `final_report.json` exactly. One does not:

| Model | Field | Spec | JSON |
|---|---|---:|---:|
| MobileNetV2 | Val macro-F1 | 0.8815 | **0.8849** |

0.8815 is MobileNetV2's *test accuracy*, repeated into the validation column. MobileNetV2 is
last on validation macro-F1 either way, so no selection or ranking claim is affected.

The spec's calibration table (§2) and ablation table (§4) both reproduce exactly.

## GFLOPs — omitted, not estimated

`stageD_complexity.json` carries `gflops: null` and
`gflops_note: "fvcore unavailable (ModuleNotFoundError)"` for all 50 runs. `torch` is not
installed on the machine doing this build either, so the measurement cannot be repeated
here. Per `MANUSCRIPT_SPEC.md` §6 the column is **omitted**, and
`tables/complexity.tex` states why in its note. No analytic estimate is presented as a
measurement anywhere.

## New evidence generated for the long format

Three sections of the 19-section target had no artefact behind them. Rather than write
around the gap, each now has a script and a JSON:

| Script | Output | Feeds |
|---|---|---|
| `experiments/classical_baseline.py` | `reports/stageG_classical.json` | §8 Classical ML Benchmark |
| `experiments/shortcut_probe.py` | `reports/stageH_shortcut.json` | §12 Dataset Contribution Analysis |
| `experiments/dataset_stats.py` | `reports/dataset_stats.json`, `manifests/` | §3 Dataset |

The **VLM family was deleted, not invented** — `AQUANET_Q4_PLAN.md` dropped it and Stage P
contains no CLIP run. No CLIP row appears anywhere.

`shortcut_probe.py` independently reproduces `AUDIT.md` finding F1. AUDIT reported a
single-seed metadata-only accuracy of 68.85% and macro-F1 0.577 against a 35.13% majority
baseline; the five-seed version gives **0.6918 ± 0.0013** accuracy and **0.5822** macro-F1
against a majority baseline of **0.3513**. `dataset_stats.py` independently reproduces the
other half of F1: **2,645 of 2,799** D1 images sit in a resolution group spanning more than
one split.

## Still open — author action required

1. **`\author{}` is still a placeholder.** Author names, affiliations and ORCIDs cannot be
   invented and are not in any artefact in this repository. `main.tex` carries a `\todo`.
2. **The Kaggle provenance question is unanswered.** `MANUSCRIPT_SPEC.md` §5 asks whether the
   Kaggle account `vasundharadixit1826` (source of D1, via
   `real-and-ai-data`) is co-author Vasundhra Dixit. The manuscript currently states the
   provenance without asserting any relationship. If the accounts are the same person, the
   data-availability statement should say so explicitly.
3. **Zenodo DOI.** The protocol section and data-availability statement carry a `\todo` for
   the archived DOI, which cannot be minted until the public repository is tagged.

---

## Public repository build (`PUBLIC_REPO_SPEC.md`)

Built at `aquanet-public/` as a fresh repository with a single commit. No history was
carried over from the working repository, whose history contains data paths, superseded
claims and the phase-3 defects.

**Consolidation.** `phase3_pipeline.py` + `phase4_pipeline.py` + `phase4_helpers.py`
became one `train.py`; `eval_suite.py` + `eval_transfer.py` + `explain.py` became one
`evaluate.py` with nine `--task` values. The items the spec required be kept are kept, with
their comments intact: `MatchedNeck` and its parameter arithmetic, the OOM handler that
deliberately refuses to retry at a smaller batch, the `wilcoxon_note`, and the `--stage`
filter with its explanation of why pooling stages corrupts the headline means.

**Verified, not asserted.** On a machine with no GPU, no images and no checkpoints:

| Check | Result |
|---|---|
| `aggregate.py` vs shipped `final_report.json` | identical to the last bit: 10 summary rows, 27 paired tests, 9 McNemar p-values |
| `evaluate.py --task abstention` vs shipped `abstention.json` | identical, largest numeric difference 0.0 across 50 runs |
| `scripts/make_tables.py` vs `paper_final/tables/` | identical, all 18 tables in both `.tex` and `.csv` |
| `evaluate.py --task binary` vs shipped `binary.json` | identical except `brier` (~6e-10 drift from a different scikit-learn build; not reported in the paper) |
| `MatchedNeck` docstring arithmetic | verified in code: 3,407,872 conv weights and 3,072 BN affine parameters, exactly matching MSRB |

**Two defects found while building it**, both now fixed in the public repository:

1. `getattr(np, 'trapezoid', np.trapz)` evaluates its default eagerly, so on NumPy ≥ 2 --
   where `trapz` was removed -- it raised `AttributeError` before the fallback could be
   reached. This broke `evaluate.py --task abstention` and `--task deletion` outright. Found
   by actually running the entry points rather than only importing them.
2. `.gitignore` initially carried a bare `runs/` rule, which would silently have excluded
   `results/runs/` -- all 222 per-run records, the evidence base the whole release exists to
   ship.

**GFLOPs remain omitted.** `torch` was not installed on the machine that produced the
original runs' complexity measurements either, so re-measuring was not possible; the column
is absent rather than estimated.

**`requirements.txt` is split into two blocks by evidence.** The analysis pins are exact and
verified to reproduce every table. The training pins are exact only where a run record
actually captured them -- `python 3.10.20`, `torch 2.6.0+cu124`, CUDA 12.4, A100-SXM4-40GB.
The `env` block never recorded the torchvision or timm versions, so those are given as lower
bounds with a note. Inventing a pin nobody observed would be a fabricated claim about the
environment.

**Still required before tagging `v1.0-submission`:** author names in `CITATION.cff` and in
`main.tex`, then the Zenodo archive, then the DOI into the manuscript's data-availability
statement and protocol section.
