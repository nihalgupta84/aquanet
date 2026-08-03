# PUBLIC_REPO_SPEC.md — build `aquanet` (public)

A clean, citable, code-and-results repository. **No internal docs, no archived manuscripts,
no data, no checkpoints.** This URL goes in the paper.

Build as a **fresh repo with no git history**. Copy files in; do not `git clone` the
working repo — its history carries data paths, superseded claims, and the phase-3 defects.

**Prerequisite: finish GATE F in `MANUSCRIPT_SPEC.md` first.** Publishing a figure script
with hardcoded numbers invites exactly the reviewer question you do not want.

---

## 1. TARGET STRUCTURE

```
aquanet/
├── README.md
├── LICENSE                      # MIT or Apache-2.0
├── CITATION.cff
├── requirements.txt             # FROZEN, exact pins
├── .gitignore
│
├── aquanet/
│   ├── models/
│   │   ├── aquanet.py           # AquaNetV3 + MSRB + CSAB + MatchedNeck
│   │   ├── baselines.py         # timm wrappers (TIMM_ALIAS map lives here)
│   │   └── classical.py         # only if GATE E was run
│   ├── data/
│   │   ├── dataset.py
│   │   └── transforms.py        # see §6 — do NOT silently fix the resize
│   └── utils/
│       ├── metrics.py           # incl. ECE, temperature scaling
│       ├── seed.py
│       └── gating.py            # SoftProbabilisticGating
│
├── train.py                     # ONE clean trainer (§2)
├── evaluate.py                  # ONE clean evaluator (§2)
├── select_lr.py                 # per-model LR selection, --table for the paper appendix
├── aggregate.py                 # statistics + final report
│
├── scripts/
│   ├── reproduce.sh
│   ├── make_tables.py
│   └── make_figures.py          # de-hardcoded version of generate_final_paper_figures.py
│
├── results/                     # ALL result JSONs — the evidence base
│   ├── final_report.json
│   ├── calibration.json
│   ├── abstention.json
│   ├── binary.json
│   ├── corruptions.json
│   ├── zeroshot.json
│   ├── adapt_*.json
│   ├── complexity.json
│   ├── deletion.json
│   └── runs/                    # per-run metrics + config + env (git_commit included)
│
├── predictions/                 # per-image labels, probs, preds — all Stage P runs
├── figures/                     # every generated figure (PDF + PNG)
├── tables/                      # every generated table (.tex + .csv)
└── manifests/                   # split file lists + MD5 + perceptual hashes
```

Shipping `results/`, `predictions/`, `figures/` and `tables/` is the strongest reviewer
signal available: anyone can recompute every statistic in the paper without a GPU or the
images.

---

## 2. SIMPLIFY THE PIPELINE

Consolidate `phase3_pipeline.py` + `phase4_pipeline.py` + `phase4_helpers.py` into one
**`train.py`**, and `eval_suite.py` + `eval_transfer.py` + `explain.py` into one
**`evaluate.py`**.

`train.py` implements the **Phase 4 protocol only** (the repaired one, per
`AQUANET_Q4_PLAN.md` §1 defects D1–D6):
- checkpoint on validation macro-F1, never NLL, never test
- one loss applied identically to every model
- separate parameter groups for pretrained trunk vs fresh modules, warmup + cosine
- class balance applied once, weights normalised to mean 1
- head and neck as independent axes (so the Stage B factorial is constructible)
- per-image predictions written every run

```
python train.py --model aquanet --head flat --msrb off --csab off --seed 7
python train.py --model swin_tiny --seed 7
```

`evaluate.py` takes `--task calibration | abstention | binary | corruptions | zeroshot |
adapt | gradcam | deletion | complexity`.

**Keep, and keep the comments on:**
- `MatchedNeck` — the parameter-matched control the ablation depends on. Its docstring
  works out the exact parameter arithmetic; that docstring is a reviewer answer.
- The OOM handler that deliberately refuses to retry at a smaller batch.
- The `wilcoxon_note` explaining why n=5 cannot reach p<0.05.
- The `--stage` filters in `eval_suite` / `eval_transfer` and the comment explaining why
  pooling stages corrupts the headline means.

**Drop:** `phase3_pipeline.py`, `backfill_predictions.py`, `generate_phase3_figures.py`,
`benchmark_model_complexity.py`, `finetune_real_data.py`, `generate_paper_visuals.py`,
`build_corpus_bib.py`, the old root `train.py` / `test.py` / `inference.py`.

**Honesty requirement.** The paper's AquaNet-Evolution section reports phase-3 numbers that
this simplified `train.py` cannot reproduce, because it implements the *repaired* protocol.
Add to the README:

> `train.py` implements the final (Phase 4) protocol. The earlier-protocol results
> discussed in the AquaNet Evolution section were produced by a superseded pipeline whose
> defects are documented in that section; those per-run metrics are preserved in
> `results/runs/` and are not re-runnable from this code.

That sentence is what keeps the simplification honest. Do not omit it.

---

## 3. EXCLUDE — hard rules

- `data/` — **never.** D1 derives from a third-party Kaggle upload mixing real and
  AI-generated imagery. Redistribution creates a licensing problem.
- `checkpoints/`, any `*.pth` / `*.pt` — too large, and they encode the training data.
- `paper_v1/`, `paper_v2/`, `paper_v3/` — archived and claim-contradicted.
- **`paper_v1/references.bib`** — fabricated entries. Under no circumstances.
- `AUDIT.md`, `RESEARCH_PLAN.md`, `AQUANET_Q4_PLAN.md`, `PHASE3.md`,
  `PUBLICATION_STRATEGY.md`, `project.md`, `project_progress.md` — internal planning. Their
  *findings* belong in the paper; the documents do not belong in the repo.
- `paper_final/` — optional. Ship it only after author names are in and the venue's
  preprint policy allows it. Never ship `main.pdf` with `\author{Anonymous Authors}`.
- `logs/`, `wandb/`, `__pycache__/`, `.ipynb_checkpoints/`

The working repo's `.gitignore` already covers `data/`, `checkpoints/`, `logs/`, `wandb/`,
`*.pth`, `*.pt`, `*.ckpt`. Carry it over and add the doc exclusions.

Pre-commit verification:
```bash
grep -rn "workspace/notebooks" .        # expect empty
grep -rn "@article\|@inproceedings" . --include=*.md   # expect empty
find . -name "*.pth" -o -name "*.pt"    # expect empty
```

---

## 4. FREEZE THE ENVIRONMENT

`requirements.txt` currently uses loose lower bounds. Replace with exact pins from the
machine that produced the results (`pip freeze`, then trim to direct dependencies).

Add `fvcore` **only if** GFLOPs are reported — `evaluate.py --task complexity` writes
`gflops: null` plus a `gflops_note` when it is missing, and a null column in a paper is
worse than none.

Record torch version, CUDA version and GPU model in the README. Every run JSON already
carries an `env` block with `git_commit` and `git_dirty`.

---

## 5. README.md — required content

**Status line, first thing on the page:**

> Research code for a controlled study of tuning-protocol effects in visual water-condition
> classification. The proposed AquaNet architecture does **not** outperform the strongest
> baselines on macro-F1 — it ranks 9th of 10. It attains the lowest uncalibrated expected
> calibration error of the ten models evaluated, with a fitted temperature of 1.098 against
> 1.591–2.753 for all baselines, at 7.61M parameters.

State it up front. A reviewer discovering the gap between "proposed model" and "9th place"
on their own reads it as spin; stating it reads as rigour.

Then:
1. **Headline tables** — the ranking table and the calibration table, identical to the
   paper's. A mismatch between repo and paper is exactly what reviewers check.
2. **Data availability** — no images shipped. State the Kaggle source and its licence, that
   D1 contains AI-generated imagery, how D2/D3 were collected, and point to `manifests/`
   (file lists + MD5 + perceptual hashes) so splits are reproducible without redistribution.
3. **Reproduction** — `scripts/reproduce.sh`, in order: Stage T LR sweep → `select_lr.py` →
   5-seed finals → Stage B factorial → evaluation tasks → `aggregate.py` → tables → figures.
4. **Known limitations** — 427-image test set, 146-image D3 (14–26 per class),
   source-label confounding (a metadata-only classifier reaches 68.85% against a 35.13%
   majority baseline), label noise, aspect-ratio distortion in preprocessing, no field
   validation. Summarise from the audit; do not ship the audit.
5. **Figure gallery** — inline the key figures, especially the fitted-temperature chart.

---

## 6. CODE FIXES BEFORE PUBLISHING

- **De-hardcode `make_figures.py`** (GATE F in `MANUSCRIPT_SPEC.md`). This is the one
  blocking item.
- `transforms.py` resizes to 224×224 without preserving aspect ratio, making source aspect
  ratio a distortion cue. **Do not silently fix it** — the published results used it. Add a
  code comment documenting the issue and pointing to the paper's Limitations section.
- Replace hardcoded absolute paths with `ROOT`-relative ones.
- Run every entry point with `--help` after restructuring to confirm imports survived.
- Re-run `aggregate.py` inside the new repo and diff against the paper's tables. They must
  match to the last decimal.

---

## 7. CITABILITY

- Tag `v1.0-submission`.
- Archive the tag on Zenodo to mint a DOI. **Cite the DOI, not the GitHub URL** — GitHub
  URLs are not archival and some journals reject them as a sole reference.
- `CITATION.cff` with authors in manuscript order.
- In the paper's protocol section: *"Code, per-run metrics, per-image predictions, and all
  figure and table generation scripts are available at <DOI>."*

---

## 8. FINAL CHECK

```bash
git log --oneline | wc -l               # expect 1
du -sh .                                # expect < ~100 MB
python -c "import json,glob; [json.load(open(f)) for f in glob.glob('results/**/*.json',recursive=True)]"
python train.py --help && python evaluate.py --help && python aggregate.py --help
python scripts/make_figures.py && python scripts/make_tables.py   # must run from JSON alone
```

Then set the working repo `nihalgupta84/aquanet` back to **private**. Only the clean repo
is public.
