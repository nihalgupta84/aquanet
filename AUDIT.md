# Independent Audit of the IrrigWater-7 / AquaNet Research Plan

Date: July 31, 2026
Scope: audit and cross-check of `RESEARCH_PLAN.md` against the data, code, results, and literature corpus actually present in this repository. Hypothesis generation only — no changes were made to the pipeline, data, or manuscripts.

> **Superseded by [`AQUANET_Q4_PLAN.md`](AQUANET_Q4_PLAN.md) (July 31, 2026). Two corrections; read them before acting on anything below.**
>
> **1. §6 Gate 1 is wrong — do not execute it.** It recommends re-splitting D1 group-disjoint. That is not constructible: class and source are near-collinear in D1 (`algae` is 933/968 one drone-video source; `foam` has 29 real images across 4 sources; `turbid` has 50 across 7), so splitting by source empties several classes from train or test.
>
> **2. Finding F7 / §0's "the ablations refute C2" is wrong — retracted.** Re-aggregating all 15 completed result files (`phase3_summary.json` was stale, written before three runs finished) shows AquaNet `no_msrb` at **0.8774** test accuracy beats ResNet50 (**0.8611**) and MobileNetV2 (**0.8603**) on **all three seeds**, and AquaNet `flat` reaches **0.8899** at seed 42. Only the hierarchically-gated `full` variant loses. `no_msrb` and `no_csab` were never run under the flat head and never against a parameter-matched control, so MSRB and CSAB remain untested rather than refuted. The dominant effect in the study is the head, not the blocks. See `AQUANET_Q4_PLAN.md` §0, §1-D5 and §2.
>
> Findings F1–F6 (metadata/source confounding, D1 provenance, corpus gap) are unaffected and remain the basis for the claim boundaries in `AQUANET_Q4_PLAN.md` §4.

---

## 0. Verdict

**The plan is procedurally sound and substantively unexecutable in its current form.**

`RESEARCH_PLAN.md` is a good scientific process document: the evidence gates, the ban on test-driven selection, the calibration/statistics requirements, and the explicit "no unsupported claims" rules are all correct and are the right response to what went wrong in `paper_v2`. The problem is not the process. The problem is that the process is aimed at validating three claims that the underlying data cannot support, and one of the plan's own gates (step 3, "Freeze IrrigWater-7 identity, manifests, provenance") is **impossible to satisfy** with what exists on disk.

Three findings dominate everything else:

| # | Finding | Status |
|---|---|---|
| **F1** | Class label is confounded with image source. A classifier using **only file metadata and no pixels** reaches **68.85% accuracy / 0.577 macro-F1** on the D1 test set (majority baseline 35.1%). Source groups are split randomly across train/val/test: 2,645 of 2,799 images sit in a resolution group that spans more than one split. | **Verified** |
| **F2** | D1 is not an original dataset. It is the public Kaggle dataset `vasundharadixit1826/real-and-ai-data` (the local path is a symlink into `~/.cache/kagglehub/...`), whose own title states it mixes real and AI-generated images. Roughly a third of D1 sits in two blocks with generative signatures. | **Verified (origin); high-confidence (AI content)** |
| **F3** | AquaNet's own ablations refute AquaNet. Removing MSRB improves accuracy on **all three seeds** (+1.80 pts mean, and it is the *smaller* model: 7.65M vs 10.53M params). Replacing the hierarchical soft-gating head with a flat head gives the best single run in the entire study (88.99%). Every "novel" component is neutral or harmful. | **Verified** |

Everything the project has been fighting for weeks — "many models we proposed all are getting failed", "sometimes problem came with data leakage, sometimes with the model" — has a single root cause. It is not the models. **The dataset has a large, trivially learnable source-identity component and a small, genuinely hard visual component. Architecture work cannot move either one.** That is exactly the signature you observe: every family converges to 85–88% within seed noise, and real-domain performance collapses to 39%.

Probability the current plan reaches a Q1 acceptance as scoped: **~5%**. Alternative paths that reach 35–55% are given in §6.

---

## 1. What was verified, and how

All checks below are reproducible; commands are in Appendix A.

### F1 — Source–label confounding and group leakage (the core defect)

D1 images cluster into 181 exact-resolution groups. Resolution is a proxy for acquisition source (one scrape, one video, one generation run). The largest groups are near-pure in class:

| Resolution | N | Class composition | Split spread (train/val/test) |
|---|---|---|---|
| 1280×960 | 936 | **algae 933**, clean 3 | 653 / 140 / 143 |
| 512×512 | 474 | foam 188, turbid 186, debris 100 (no other class) | 333 / 69 / 72 |
| 1024×559 | 452 | all 7 classes | 321 / 64 / 67 |
| 1000×1000 | 73 | uncertain 60, oil 13 | 65 / 5 / 3 |
| 3565×2674 | 22 | debris 19, turbid 3 | 12 / 3 / 7 |
| 3614×2711 | 22 | debris 19, turbid 3 | 13 / 7 / 2 |

Two things follow.

**(a) The split is group-random, not group-disjoint.** Every large group is divided ~70/15/15 — the exact global split ratio. 2,645 / 2,799 images (94.5%) belong to a group spanning more than one split. For the 22-image groups at 3565×2674 and 3614×2711 — resolutions that can only come from a single camera or a single shoot — sibling images are sitting in train and test simultaneously. The MD5/dHash audit cannot see this: siblings from the same video seconds apart are not near-duplicates at Hamming ≤5, but they are not independent either.

**(b) The confound is measurable without touching a single pixel.** A random forest over `(width, height, aspect, area, file size, bytes-per-pixel, JPEG quantization-table signature)` — no image content whatsoever — scores:

```
METADATA-ONLY classifier          test accuracy 0.6885   macro-F1 0.5770
  algae   P 0.960  R 0.960  F1 0.960   (n=150)
  uncertain P 0.743 R 0.786 F1 0.764   (n=70)
majority-class baseline           0.3513
```

Compare against the deep models (3-seed means, same test set):

| Model | Accuracy | Macro-F1 | algae F1 |
|---|---|---|---|
| **Metadata only, zero pixels** | **0.6885** | **0.5770** | **0.960** |
| MobileNetV2 (2.2M) | 0.8603 | 0.8292 | ~0.97 |
| ResNet50 (23.5M) | 0.8610 | 0.8277 | 0.966 |
| AquaNet v3 full (10.5M) | 0.8595 | 0.8233 | 0.980 |
| AquaNet v3 flat (seed 42) | 0.8899 | 0.8586 | 0.990 |

The whole deep-learning contribution over a no-pixel baseline is ~17 accuracy points. And `algae`, which is 35% of the test set, is solved to F1 0.96 **by metadata alone** — the CNNs' 0.98 on the largest class is not evidence of visual algae recognition.

*Precision of the claim:* this proves **source–label confounding and group leakage in the split**. It does not by itself prove the CNNs read resolution — they see 224×224 tensors. But the mechanism is available to them: `get_transforms` uses `Resize((224,224))`, a non-aspect-preserving squash, so a 4:3 source is anisotropically distorted and a 1:1 source is not; and same-source images share color grading, compression artifacts, framing, and optics regardless. The decisive confirmation is the source-group-disjoint retrain in §6, Gate 1.

### F2 — Dataset composition and origin

`data/cleaned_water_dataset` derives from `/workspace/notebooks/vasundhra/data/real-and-ai-data/444dataset`, which is a symlink to:

```
/root/.cache/kagglehub/datasets/vasundharadixit1826/real-and-ai-data/versions/1/444dataset
```

So D1 is a **third-party-published Kaggle dataset whose name declares that it mixes real and AI-generated imagery**. (If `vasundharadixit1826` is the project's own collaborator, ownership is fine — but it is already public under that name, which materially weakens "we introduce a new dataset" and makes the AI content a disclosed fact rather than a discoverable one.)

Visual inspection of the three dominant blocks:

- **1280×960 (936 images, 33% of D1, 94% of the algae class):** aerial/drone frames from a small number of flights over the same water bodies — continuous viewpoint, altitude, and lighting across frames. One sample labelled `algae` shows open blue ocean with clouds and no visible bloom. This is video-frame leakage at scale plus label noise.
- **512×512 (474 images, 17%):** strong generative signature — exactly Stable-Diffusion-native resolution, uniform across all 474 files, zero EXIF, only three classes present (one generation run per class), soft-lobed foam with implausible geometry, no sensor noise.
- **1024×559 (452 images, 16%):** non-camera aspect ratio (1.83:1 matches no standard sensor), symmetric wide-angle canal compositions, physically implausible canal geometry in places, spans all 7 classes as if from one prompt template varied by class.

Best estimate of D1: **~33% one drone-video source, ~33% likely generated, ~33% heterogeneous real web imagery.** Only that last third is what the paper claims the whole dataset is.

D2 (213) and D3 (146) are uniformly 450×450 with zero EXIF — every image resized by the scraper. So the reported D1→D3 "domain gap" is partly a **preprocessing artifact**, not only a domain shift.

### F3 — The ablations refute the architecture

Current standings from `phase3_results/` (the 4 runs still on the GPU will extend the last two rows):

| Config | Seeds | Accuracy | Macro-F1 | Params |
|---|---|---|---|---|
| aquanet_v3 / **flat** | [42] | 0.8899 | 0.8586 | 10.53M |
| aquanet_v3 / **no_msrb** | [7,21,42] | **0.8774 ±0.0014** | 0.8421 ±0.0019 | **7.65M** |
| aquanet_v3 / no_csab | [42] | 0.8665 | 0.8319 | 10.50M |
| resnet50 / full | [7,21,42] | 0.8610 ±0.0141 | 0.8277 ±0.0152 | 23.52M |
| mobilenetv2 / full | [7,21,42] | 0.8603 ±0.0118 | 0.8292 ±0.0128 | 2.23M |
| **aquanet_v3 / full** | [7,21,42] | **0.8595 ±0.0107** | 0.8233 ±0.0157 | 10.53M |

Removing MSRB, per seed: +0.0281 (s7), +0.0047 (s21), +0.0211 (s42) — **positive on every seed**, mean +1.80 pts, on a model with 27% fewer parameters. (Paired t on n=3 gives p=0.12, but n=3 has essentially no power; the consistent sign across seeds with σ=0.0014 is the informative part.)

The full model is also the **least stable** configuration (σ=0.0107 vs 0.0014 for no-MSRB) — the soft-gating path is adding variance, not robustness.

The proposed architecture is currently in last place among its own variants, behind a 2.2M-parameter MobileNetV2. There is no model contribution to write up.

### F4 — Protocol defects in `experiments/phase3_pipeline.py`

These are independent of the data problem and each one is individually disqualifying for a fair-benchmark claim.

1. **The baselines and the proposed model are trained with different losses.** Line 100–104: AquaNet gets weighted cross-entropy (or a 0.5/0.5 gated-NLL + flat-CE mix); ResNet50 and MobileNetV2 get **weighted focal loss (γ=2)**. The comparison is not protocol-matched, which violates the plan's own §7 fairness requirement. Any ranking derived from it is uninterpretable.

2. **Class balancing is applied twice.** `loaders()` uses `WeightedRandomSampler` with 1/n weights (line 66), *and* the loss applies 1/n class weights (line 89). The sampler already makes the batch distribution uniform; the loss weights then re-apply the same correction on top. Minority classes are over-weighted by roughly the imbalance ratio (~6.5×) **beyond** balance. This is visible in the results: `oil` has precision 0.50 with recall 0.875 — the model is massively over-predicting the rarest class. It also destroys calibration, which then propagates into the ECE/NLL numbers the plan wants to report.

3. **`lr=1e-3` AdamW on all parameters, including a pretrained backbone, with no warmup or discriminative learning rates.** This is 10× the normal fine-tuning rate and will damage ImageNet features early in training. It is the most likely explanation for the historic DenseNet121 baseline at 74.24% — a number that cannot be reconciled with DenseNet-backbone variants scoring 87–89% here. **Every baseline in this project is plausibly under-tuned, so the entire benchmark ranking is unreliable in both directions.**

4. **The "flat" variant is a loss ablation, not a head ablation.** All four variants build and train `flat_head`. `full`/`no_msrb`/`no_csab` train on a 50/50 mix of gated NLL and flat CE and then *evaluate through the gated path*; `flat` trains on flat CE alone and evaluates the flat head. So `flat` vs `full` measures "does hierarchical soft gating help?" — answer: it costs 3.3 points — and the full model is additionally handicapped by half its gradient signal going to a head it does not use at inference.

5. **Checkpoint selection uses weighted validation NLL** (line 112), inheriting the double-balancing distortion, while `RESEARCH_PLAN.md` §8 mandates selection on validation macro-F1. The plan and the code disagree.

6. **`Resize((224,224))` destroys aspect ratio** — the mechanism by which source geometry becomes a pixel-level cue (see F1).

7. **`wilcoxon` on n=3** (line 140) cannot return p < 0.25 and should not be reported. Latency is measured as batched throughput at batch 16 and labelled "ms per image".

### F5 — The literature corpus does not cover this problem

All 25 papers in `corpus/extracted/` were scanned. **Not one studies RGB-image classification of water-surface contamination type.** Zero mentions of "foam", "oil sheen", or "floating debris" across the entire corpus. The corpus is: IoT/electrochemical sensing (turbidity, pH, EC, DO), tabular WQI/SAR/ESP regression, satellite remote sensing, and agriculture-ML reviews.

Consequences:

- `RESEARCH_PLAN.md` §16 requires a "literature taxonomy" table and §18 forbids SOTA claims "without comparable external evidence". There is currently **no comparable external evidence in the corpus at all**, because the corpus is about a different measurement modality.
- `paper_v3/references.bib` contains **6 entries**, all generic architecture papers (DenseNet, ResNet, MobileNetV2, CBAM, focal loss, calibration). There is not a single domain citation in the manuscript.
- Worse, the corpus arms the reviewer against the paper. It establishes that this field measures water quality with cheap physicochemical probes producing calibrated, quantitative, actionable values. A reviewer at IEEE TIM will ask: *what irrigation-relevant quantity does "foam vs. algae vs. debris" measure, and why is it preferable to a $20 turbidity sensor?* The plan has no answer to that question, and §13's instruction to "frame the camera/model as a visual measurement system" does not supply one — framing is not a measurement.

### F6 — The class taxonomy contains a non-class

`uncertain` (459 images, 16% of D1) is an **annotation state, not a water condition**. Three problems compound:

- It is filed under `contaminated/uncertain` in the directory tree, so `water_dataset.py` assigns it `binary_label = 1`. The hierarchy therefore asserts "the annotator could not tell ⟹ the water is contaminated", which is false by construction.
- Its global color statistics are near-identical to `clean` (project_progress §3: H 83.0 vs 85.3), so it is not separable by the very features the architecture was designed around.
- It is the class where metadata-only prediction is second-strongest (F1 0.764) — i.e. "uncertain" largely encodes *which source the image came from*, not any property of the water.

This is a strong candidate explanation for why the hierarchical head loses to the flat head: the gating is being forced to route genuinely-ambiguous images through P(contaminated). The correct treatment is abstention/selective prediction, not a seventh class.

### F7 — The release package in §5 cannot be built

`RESEARCH_PLAN.md` §5 requires manifests with: source type, source URL or device, video/source group, frame/timestamp, natural/generated status, license, annotation method, and annotator agreement.

**None of these exist.** D1 filenames are `algae_train_0000.jpg`; the upstream Kaggle files are `4240.jpg`. All provenance was destroyed during cleaning. There is no annotation protocol, no second annotator, and therefore no agreement statistic to report. Licenses for the scraped and generated content are unknown.

There is one partial recovery route: `/workspace/collected_all_video_urls/collected_video_urls.jsonl` (2.1 MB) and `/workspace/water_contamination_video_sources.json` define exactly the right schema (`source_url`, `license`, `creator_or_channel`, `frame_sampling_fps`, `start_time_seconds`…). The collection infrastructure was designed correctly; the link between those records and the delivered image files was not preserved. Re-deriving it would require re-running frame extraction from the source videos, not repairing the current files.

**Plan step 3 is therefore a blocker, not a task.** No amount of GPU time advances it.

---

## 2. Cross-check of `RESEARCH_PLAN.md`, section by section

| Plan item | Status | Reason |
|---|---|---|
| §1 thesis: "dataset-and-model study" | **Both halves fail** | Dataset is third-party + AI-mixed (F2); model loses to its own ablations (F3) |
| §2 IrrigWater-7 identity, name freeze | **Blocked** | Cannot name and release a dataset whose provenance is gone and whose origin is a public Kaggle upload (F2, F7) |
| §3 candidate titles | Premature | All three titles assert a dataset contribution |
| C1 named multi-domain dataset | **Blocked** | F2, F7 |
| C2 validated model | **Refuted so far** | F3 |
| C3 cross-family benchmark | **Invalid as run** | F1 (confounded labels) + F4.1/F4.3 (mismatched losses, mis-tuned LR) |
| C4 domain generalization | **Partly artifactual** | D2/D3 uniformly 450×450; gap is preprocessing + source shift, not only domain (F2) |
| C5 model understanding | Salvageable | Ablations are informative *as a negative result*; explainability on a shortcut-driven model would visualize the shortcut |
| §5 release package | **Impossible as specified** | F7 |
| §6 "adjudicate 698 dHash candidates" | **Necessary but insufficient** | dHash≤5 finds re-encodes (142 pairs at distance 0!) but is blind to same-video sibling frames, which is the dominant leakage (F1) |
| §7 fairness requirements | **Violated by current code** | F4.1, F4.3, F4.5 |
| §8 validation-only selection | Correct in principle, **contradicted by code** | Code selects on weighted val NLL, plan says val macro-F1 (F4.5). Also moot until the split is fixed |
| §9 five-seed statistics | Correct but **not the bottleneck** | More seeds on a leaky split measure the leak more precisely |
| §10 metrics | Sound | — |
| §11 ablation program (~30 configs × 5 seeds) | **Highest-cost, lowest-value item in the plan** | Runs ≈150 GPU-jobs to resolve differences smaller than the confound. Do not start before Gate 1 |
| §12 dataset/domain experiments | **Now the most valuable section** | "Natural-only vs generated-only" is no longer optional — it is the headline experiment (F2) |
| §13 robustness / TIM framing | Framing ≠ measurement | F5: no reference instrument, no calibrated quantity |
| §14 explainability | Correct to demand genuine methods | But Grad-CAM on a shortcut model shows the shortcut; run it *after* Gate 1 |
| §15/§16 figures & tables | Fine as a checklist | Contents depend entirely on which path is taken |
| §17 19-section manuscript | Over-scoped | 19 sections is a monograph; most Q1 venues want 8–12 |
| §18 evidence rules | **Keep verbatim** | This is the best part of the document and it is what caught the paper_v2 problems |
| §19 execution order | **Wrong first step** | Step 1 spends GPU time on a comparison that Gate 1 invalidates; step 3 is blocked |

**Net:** roughly 60% of the plan is well-designed work that is currently pointed at an invalid substrate; ~20% (release package, dataset identity) is blocked outright; ~20% (evidence rules, metrics, statistics discipline) is directly reusable in any path.

---

## 3. Root-cause hypotheses, with falsifiable predictions

Stated so each can be killed cheaply.

**H1 — Shortcut saturation.** D1 accuracy decomposes into a large source-identity component (~69%, learnable without pixels) and a small genuinely-visual component. Architectures compete only over the residual, which is why all six configurations land within ±1.5 points.
> *Predicts:* under source-group-disjoint splits, D1 test accuracy falls from ~86% to **60–72%**; `algae` F1 falls from 0.98 to **below 0.85**; the spread between MobileNetV2, ResNet50, and AquaNet **shrinks further**, not grows.
> *Killed if:* accuracy holds above ~82% under group-disjoint splits.

**H2 — Synthetic-to-real is the real gap.** The 39% zero-shot on D3 is mostly a real-vs-generated style shift, not a lab-vs-field shift.
> *Predicts:* training on the **natural subset only** of D1 (excluding the 512×512 and 1024×559 blocks) transfers to D3 *better* than training on all of D1, despite ~33% less data.
> *Killed if:* natural-only training transfers equal or worse.

**H3 — The hierarchy is fighting a mislabeled taxonomy.** Soft gating loses to a flat head because `uncertain` is forced through P(contaminated).
> *Predicts:* removing `uncertain` (6-class) or moving it to an abstention channel shrinks or reverses the flat-vs-hierarchical gap.
> *Killed if:* flat still beats hierarchical by ~3 points at 6 classes.

**H4 — Every model here is mis-trained.** lr=1e-3 on a pretrained backbone plus double class balancing suppresses all models and inflates minority-class over-prediction.
> *Predicts:* with lr=1e-4 + warmup, sampler **or** loss weighting (not both), all models gain ~2–4 points, `oil` precision rises from 0.50 toward ~0.75, ECE improves, and the ranking may change.
> *Killed if:* corrected training changes nothing.

**H5 — There is no measurand.** "algae / foam / oil / debris / turbid / clean / uncertain" from a single RGB frame is not a measurement of irrigation water quality; it is scene description. The corpus (F5) shows the field wants calibrated physicochemical values.
> *Predicts:* reviewers at any instrumentation venue will ask for correlation against a reference instrument, and there is no answer.
> *This one cannot be killed with GPU time.* It is a scoping decision, and it determines which venue is reachable.

---

## 4. Why the current results look the way they do

A single consistent story explains every anomaly in the project log:

- **90.40% (old) → 85.95% (Phase 3), same model.** The old number came from a pipeline with byte-identical duplicates across splits; Phase 3 removed those but left group leakage. Both are inflated; the newer one is less so.
- **DenseNet121 at 74.24% but DenseNet-backbone AquaNet variants at 87–89%.** Not an architecture effect — an artifact of lr=1e-3 destroying pretrained features (H4) plus differing loss functions (F4.1).
- **All families converge to 85–88%.** H1: they are all solving the same ~69% source puzzle plus the same hard residual.
- **Removing components helps.** With the label largely determined by source, extra capacity in the neck fits noise. The smallest, simplest variant (no-MSRB, 7.65M) is the most stable (σ=0.0014).
- **39% zero-shot on D3, 57.5% after fine-tuning.** H2: the model learned generated-image and drone-video style. When both vanish, so does the performance. 57.5% after adaptation is still poor because there was never much transferable visual knowledge.
- **`oil` precision 0.50 / recall 0.875.** F4.2: double class balancing over-weights the rarest class ~6.5× beyond uniform.
- **`uncertain` F1 0.65–0.78 and confusable with `clean`.** F6: it is not a visual class.

---

## 5. Probability assessment

Success = accepted at a Q1 venue, roughly as scoped.

| Path | What it claims | P(Q1) | Time | Blocking dependency |
|---|---|---|---|---|
| **D. Status quo** — execute `RESEARCH_PLAN.md` as written | New dataset + superior model | **~5%** | 3–4 months | Blocked at plan step 3; C1 and C2 both unsupportable |
| **A. Reframe as a data-artifact / evaluation-protocol study** | Web-sourced environmental image datasets encode source identity so strongly that architecture comparison becomes noise; here is a protocol and a diagnostic | **35–45%** | 4–8 weeks | None — uses existing data and existing runs |
| **B. Rebuild the dataset provenance-first, then run the original plan** | Genuinely new, documented, real-only dataset + benchmark | **40–50%** | 3–5 months | Re-collection with source manifests; 2+ annotators for a κ statistic |
| **C. Pivot to visual measurement against a reference instrument** | Camera-based turbidity/index estimation with calibration and uncertainty vs. a probe | **45–55%** | 4–6 months | Paired image + sensor data; field or lab access |

**Why D is ~5% and not zero:** a determined author can always find a venue. But C1 (dataset contribution) cannot survive a reviewer typing the dataset name into Kaggle, and C2 (model contribution) is already refuted by the project's own ablation table. Executing the remaining eleven roadmap items faithfully produces cleaner evidence for a negative conclusion — which is honest and valuable, but it is Path A's paper, not Path D's.

**Why A is the recommended path:** it is the only one where the surprising, defensible result *already exists in this repository*. The metadata-only baseline took twenty minutes to produce and is the kind of finding a good reviewer respects — and, importantly, the kind a reviewer would otherwise discover and reject you for. The 18 Phase-3 runs become the "leaky-split arm" of the central experiment rather than wasted compute. Candidate venues where this is Q1: *Ecological Informatics*, *Environmental Modelling & Software*, *Pattern Recognition Letters*, or a data-centric-AI track. It is not IEEE TIM — TIM needs Path C.

**On running A and B together:** A is publishable in ~6 weeks and derisks the effort; B then becomes the follow-up paper with a dataset that can actually be released. That sequencing is strictly better than pursuing B alone, because A's protocol contribution is what makes B's dataset credible.

---

## 6. Revised roadmap

### Immediate (this week, no new GPU cost)

The four remaining Phase-3 runs (no_csab and flat at seeds 7, 21) are **worth finishing** — they complete the leaky-split ablation arm that Path A needs, and they are ~3 hours of already-committed compute. But their output must **not** be used for model selection (roadmap step 2), because the split they are measured on is invalid.

- Freeze the current 18 runs as the **"group-random split" arm** and label them as such everywhere.
- Retract the model-selection gate from the roadmap until Gate 1 closes.

### Gate 1 — Establish the true baseline (highest priority, ~1 week)

This gate decides which paper exists. Nothing else should start before it closes.

1. **Build a source-group assignment for all 3,158 images.** Cluster on (exact resolution) ∪ (dHash ≤ 10 transitive closure) ∪ (JPEG quantization-table signature) ∪ (color-histogram nearest-neighbour chains). This does not need the lost provenance — it recovers acquisition groups forensically.
2. **Label each group `natural` / `likely-generated` / `drone-video`.** Manual review of ~50 sampled group thumbnails, plus a generated-image detector as corroboration. Record the decision rule; this becomes a paper table.
3. **Re-split D1 group-disjoint**, stratified by class, and re-run the six configurations at three seeds on the new split.
4. **Report the metadata-only baseline alongside every model, on both splits.** This is the diagnostic contribution.

**Decision rule at Gate 1:** if group-disjoint accuracy lands in the 60–72% band predicted by H1, H1 is confirmed and Path A is the paper. If it holds above ~82%, H1 is wrong, the data is better than it looks, and Path B/D become viable again.

### Gate 2 — Fix the protocol (~3 days, run concurrently with Gate 1)

Independent of the data question, and required for any path:

- One loss for all models (weighted CE), or report both losses for all models.
- Class balancing **once**: sampler or loss weights, not both.
- `lr=1e-4` with warmup, or discriminative LRs (backbone 1e-4 / neck+head 1e-3).
- Checkpoint on validation macro-F1, matching plan §8.
- Aspect-preserving resize (letterbox or resize-shortest-side + centre crop) to close the geometric shortcut channel.
- Add **DenseNet121 under the identical protocol** — the one baseline the entire AquaNet story depends on and the one that is missing.
- Drop `wilcoxon` at n=3; use McNemar on paired predictions (already in plan §9).
- Relabel the latency column as batched throughput.

### Gate 3 — The Path A experiments (~3 weeks)

1. Group-random vs group-disjoint, all six configurations × 3 seeds → the confound's magnitude.
2. Metadata-only and colour-histogram-only baselines on both splits → the shortcut floor.
3. Natural-only vs generated-only vs combined training, each evaluated on D3 (tests H2).
4. 6-class (drop `uncertain`) vs 7-class, flat vs hierarchical (tests H3).
5. Corrected vs original training protocol (tests H4).
6. Grad-CAM **contrasting** the two splits — showing the group-random model attending to background/framing and the group-disjoint model attending to the water surface. This is the figure that sells the paper, and it is genuine explainability rather than decoration.

### Gate 4 — Manuscript

Retire `paper_v3`. The Path A paper is 8–10 sections, not 19. Keep `RESEARCH_PLAN.md` §18 (evidence rules) verbatim as the drafting constraint — it is the reason this audit found problems before a reviewer did.

**Related-work is a hard blocker for any path.** The corpus must be extended before drafting, with primary sources on: shortcut learning and dataset bias (Geirhos et al., Torralba & Efros), group-aware splitting and leakage in applied vision, synthetic-image contamination of training corpora, and the actual image-based water/environmental monitoring literature (HAB detection from imagery, camera-based turbidity estimation, surface-debris detection). Roughly 25–35 new papers. **None of the 25 papers currently in `corpus/` can be cited in support of the method.**

### What to explicitly not do yet

- The §11 ablation program (~30 configs × 5 seeds ≈ 150 runs). It resolves differences smaller than the confound. Deferred until after Gate 1.
- Freezing the IrrigWater-7 name, dataset card, or release policy. Blocked by F7 regardless of effort.
- Five-seed final comparisons. More seeds on an invalid split buy precision, not validity.
- Any VLM benchmark family. Adding model families to a confounded benchmark multiplies the invalid numbers.

---

## Appendix A — Reproducing the audit

```bash
cd /workspace/projects/vision/aquanet
PY=/workspace/miniconda3/envs/vision/bin/python

# F1a — resolution groups vs class, and spread across splits
$PY - <<'EOF'
from PIL import Image; import glob, collections
rows=[(f.split('/')[2], f.split('/')[-2], Image.open(f).size)
      for f in sorted(glob.glob('data/cleaned_water_dataset/**/*.jpg', recursive=True))]
cls=collections.defaultdict(collections.Counter); spl=collections.defaultdict(collections.Counter)
for s,c,z in rows: cls[z][c]+=1; spl[z][s]+=1
for z,_ in sorted(cls.items(), key=lambda x:-sum(x[1].values()))[:10]:
    print(z, sum(cls[z].values()), dict(cls[z].most_common(3)), dict(spl[z]))
print('in multi-split groups:', sum(sum(v.values()) for v in spl.values() if len(v)>1), '/', len(rows))
EOF

# F1b — metadata-only classifier (no pixels)
$PY - <<'EOF'
from PIL import Image; import glob, os, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
C=['clean','algae','debris','foam','oil','turbid','uncertain']
def load(sp):
    X=[];y=[]
    for f in sorted(glob.glob(f'data/cleaned_water_dataset/{sp}/**/*.jpg', recursive=True)):
        im=Image.open(f); w,h=im.size; q=getattr(im,'quantization',{}) or {}
        fs=os.path.getsize(f)
        X.append([w,h,w/h,w*h,fs,fs/(w*h),sum(sum(v) for v in q.values()),len(q)])
        y.append(C.index(f.split('/')[-2]))
    return np.array(X,float), np.array(y)
Xtr,ytr=load('train'); Xte,yte=load('test')
p=RandomForestClassifier(400,random_state=0,n_jobs=-1).fit(Xtr,ytr).predict(Xte)
print('acc %.4f  macroF1 %.4f'%(accuracy_score(yte,p), f1_score(yte,p,average='macro')))
print(classification_report(yte,p,target_names=C,digits=3,zero_division=0))
EOF

# F2 — dataset origin
ls -la /workspace/notebooks/vasundhra/data/real-and-ai-data/
#  -> 444dataset -> /root/.cache/kagglehub/datasets/vasundharadixit1826/real-and-ai-data/versions/1/444dataset

# F3 — current standings across all completed runs
$PY - <<'EOF'
import json, glob, collections, numpy as np
g=collections.defaultdict(list)
for f in glob.glob('phase3_results/*_seed*.json'):
    d=json.load(open(f)); g[(d['model'],d['variant'])].append(d)
for k,v in sorted(g.items(), key=lambda x:-np.mean([r['accuracy'] for r in x[1]])):
    a=np.array([r['accuracy'] for r in v])
    print('%-24s n=%d  %.4f %s  %.2fM'%('/'.join(k), len(v), a.mean(),
          '±%.4f'%a.std(ddof=1) if len(a)>1 else '       ', v[0]['params_m']))
EOF

# F5 — corpus relevance scan
cd corpus/extracted && for d in */; do f=$(find "$d" -name '*.md' | head -1); [ -z "$f" ] && continue
  printf '%-56s foam/oil/debris=%s turbidity=%s sensors=%s\n' "${d:0:54}" \
    "$(grep -ociE '\b(foam|oil sheen|floating debris)\b' "$f")" \
    "$(grep -ociE '\bturbidity\b' "$f")" \
    "$(grep -ociE '\b(pH|electrical conductivity|dissolved oxygen)\b' "$f")"; done
```

## Appendix B — Files examined

`RESEARCH_PLAN.md`, `PHASE3.md`, `project.md`, `project_progress.md`, `README.md`,
`experiments/phase3_pipeline.py`, `models/proposed/aquanet_v3.py`, `dataset/water_dataset.py`,
`dataset/transforms.py`, `utils/soft_gating.py`, all 15 files in `phase3_results/`,
`paper_v3/main_v3.tex`, `paper_v3/references.bib`, all 25 extractions in `corpus/extracted/`,
`data/` (3,158 images: dimensions, EXIF, JPEG tables, visual inspection of the three dominant blocks),
`/workspace/notebooks/vasundhra/data/` (upstream sources), `/workspace/collected_all_video_urls/`.
