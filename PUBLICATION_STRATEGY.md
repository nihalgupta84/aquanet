# Publication Strategy: Why the Baselines Beat Your Models, and What to Publish

> **WITHDRAWN (July 31, 2026). Superseded by [`AQUANET_Q4_PLAN.md`](AQUANET_Q4_PLAN.md).**
>
> §3, §4, §6 and §7 recommended publishing from `/workspace/projects/vision/water_contamination_project`. That is **not** the direction: aquanet is a standalone project and the two are not to be combined. The other project was consulted only as a possible data source and is out of scope.
>
> §1 Evidence A and §5 are also wrong on the central point. They state that no AquaNet configuration beats the baselines; re-aggregating all 15 completed runs shows AquaNet `no_msrb` beats ResNet50 and MobileNetV2 on test accuracy on all three seeds, and AquaNet `flat` is the top result in the study. The "aquanet, model-wins framing: 15–25%" row in §5 is revised to **80–85% at Q4**. See `AQUANET_Q4_PLAN.md` §0 and §5.
>
> Retained for the record only. Nothing in this file should be executed.

Date: July 31, 2026
Covers: `/workspace/projects/vision/aquanet` and `/workspace/projects/vision/water_contamination_project`
Constraints given: no new data of any kind; target Q3/Q4 journal; the paper must propose a model that wins; 4+ months available.
Supersedes: `AUDIT.md` §6 Gate 1 (see Correction below).

---

## 1. The question: "why is another model better than ours?"

**It isn't. You have been reading the wrong column.**

Across both projects, the benchmark you have been ranking models on does not measure contamination detection. It measures *which dataset an image came from*. Generic high-capacity backbones are better at that than your custom architectures, so they win that column. On every measurement where the source shortcut is unavailable, your architectures win — in one case by a factor of nearly two.

### Evidence A — aquanet

A classifier given only file metadata (width, height, file size, JPEG quantization table) and **no pixels at all** scores 68.85% accuracy / 0.577 macro-F1 on the D1 test set. Deep models score 0.86. The `algae` class — 35% of the test set — is solved to F1 0.960 by metadata alone.

Ranking on that test set: ResNet50 0.8610, MobileNetV2 0.8603, AquaNet v3 0.8595. **Those three numbers are separated by 0.0015 on a benchmark where 0.69 is free.** The "AquaNet loses" conclusion is not a fact about AquaNet.

### Evidence B — water_contamination_project (the decisive one)

The Phase 6 de-leaked test set decomposes into two source blocks:

| Source | N | Label composition |
|---|---:|---|
| `river_floating_trash_rft` | 720 | 100% contaminated |
| `phase1` | 534 | 463 clean / 71 contaminated |

So this rule — **no pixels, no model, no learning** —

> predict *contaminated* if and only if the image came from the River-Floating-Trash dataset

scores:

```
accuracy 0.9434    macro-F1 0.9409
  clean         P 0.8670  R 1.0000  F1 0.9288
  contaminated  P 1.0000  R 0.9102  F1 0.9530
```

Against your reported de-leaked results:

| Model | Reported macro-F1 | Gain over knowing only the source |
|---|---:|---:|
| CanalScopeNetV3 language | 0.9642 | **+0.023** |
| ConvNeXt-Base | 0.9622 | **+0.021** |
| Fusion | 0.9613 | +0.020 |
| CanalScopeNetV3 backbone_only | 0.9578 | +0.017 |
| CanalScopeNetV2 | 0.9508 | **+0.010** |

Every model in the project sits within 1–2 points of a rule that looks at nothing. The entire spread between "ConvNeXt wins" and "CanalScopeNetV2 loses" is 0.011 macro-F1 — **half the size of the gap between either of them and no model at all.**

### Evidence C — the field data proves it

`convnext_deleaked` scores 0.9622 on the de-leaked test and **0.3361** on 494 real field images, with contaminated recall 0.1768. Its per-subtype accuracy on real water:

| Subtype | Accuracy |
|---|---:|
| turbid | **0.0100** |
| uncertain | **0.0000** |
| algae | 0.0506 |
| foam | 0.0600 |
| debris | 0.5299 |
| clean | 0.9898 |

A model that has genuinely learned "is this water contaminated" does not get 1% of turbid images and 0% of uncertain images right. This model learned **"does this look like a floating-trash photograph"** — so it detects visible debris (0.53) and calls everything else clean (clean recall 0.9898). That is the shortcut, visible in the wild.

### Evidence D — where the shortcut is gone, your model wins

| Benchmark | ConvNeXt baseline | CanalScopeNetV3 | Δ |
|---|---:|---:|---:|
| De-leaked test *(shortcut-saturated)* | 0.9578 `backbone_only` | 0.9642 `language` | +0.006 |
| FLUX synthetic field proxy | 0.7076 | 0.7585 `full` | **+0.051** |
| HiDream cross-generator | 0.5105 | 0.5406 `color_texture` | **+0.030** |
| **Real field data (Phase 10)** | **0.3361** `convnext_deleaked` | **0.6113** `v3_language` | **+0.275** |

**On real field water your architecture nearly doubles the baseline's macro-F1.** The advantage grows monotonically as the shortcut weakens: +0.006 → +0.030 → +0.051 → +0.275.

That is a clean, honest, positive model-novelty result, and it is already in your repository at `PHASE8B_RESULTS.md` and `PHASE10_FIELD_TEST_RESULTS.md`. Nobody needed to run anything new.

### Why this happens, mechanically

Your custom branches — HSV histograms, Laplacian texture, language anchors — encode *what contaminated water looks like*: green tint, surface texture, turbidity, film. Those are weak, low-capacity, hand-specified features.

When a source shortcut is available, weak explicit features are **worse than useless** — they compete with a shortcut that a 88M-parameter ImageNet-22k ConvNeXt exploits more efficiently. When the shortcut is removed, those same features are the only thing carrying real signal, and the big backbone has nothing to fall back on.

**Your inductive biases were correct. Your evaluation was rewarding their absence.**

---

## 2. Correction to `AUDIT.md`

`AUDIT.md` §6 Gate 1 recommended re-splitting aquanet's D1 group-disjoint. **That is not constructible** and should not be attempted. Class and source are near-collinear in D1: `algae` is 933/968 one drone-video source, `foam` has 29 real images across 4 sources, `turbid` has 50 across 7. Splitting by source empties several classes from either train or test. The largest honest benchmark recoverable from D1 is roughly clean/debris/oil/turbid at ~430 images.

One finding did *not* replicate, and this is to the water project's credit: the **file-metadata** probe fires in aquanet (0.689 acc) but **fails completely** on the water project's de-leaked split (0.6308 = exactly the majority baseline, zero clean recall). Their `source_unit`-level de-leaking is real — 88 source units, **0 images in units spanning more than one split**. The residual confound there is *visual source style*, not file metadata, which is a genuinely harder problem and a milder failure.

---

## 3. Recommendation: publish from `water_contamination_project`, not aquanet

This is not close. Side by side:

| | aquanet | water_contamination_project |
|---|---|---|
| Provenance manifests | ❌ destroyed (filenames renamed) | ✅ `source_id`, `source_unit`, `source_type`, `label_source`, real paths |
| Source-aware splits | ❌ not constructible | ✅ 88 units, 0 spanning |
| Dataset licenses | ❌ unknown | ✅ 4 verified (CC BY 4.0, CC BY-NC-SA 3.0 IGO, CC0 1.0, CC BY-NC 4.0), 4 to verify |
| Real field validation | ❌ none | ✅ 494 labelled field images + adaptation |
| Model beats its own control | ❌ every ablation beats the full model | ✅ +0.006 / +0.030 / +0.051 / +0.275 |
| Deployment evidence | ❌ none | ✅ Jetson Nano latency/FPS/RSS + INT8, ~4× size reduction |
| Statistics | paired t on n=3 | ✅ bootstrap 95% CIs, paired tests, val-only threshold calibration |
| Ablation control | confounded with capacity | ✅ explicit `backbone_only` control |

aquanet cannot produce a model-wins paper from the data on disk. The water project already has one and doesn't know it.

---

## 4. The paper

**Framing:** in-distribution benchmarks for water-contamination vision are saturated by dataset-source identity; models selected on them fail in the field; an architecture with explicit color/texture/semantic priors is the fix.

This framing is honest, it is supported entirely by evidence already collected, and — critically — **it makes your model the winner** rather than requiring you to explain away a loss.

**Contributions**

1. **A source-identity ceiling diagnostic.** Report what a no-pixel source rule achieves on any benchmark before reporting model numbers. On your own de-leaked split that ceiling is macro-F1 0.9409. Cheap, reusable, and it reframes the whole literature's 0.95+ numbers. aquanet's metadata-only probe (0.689) is the second data point.
2. **CanalScopeNetV3** — ConvNeXt + HSV/Laplacian color-texture branch + frozen language-anchor attention + contrastive anchor loss, with a proper `backbone_only` control.
3. **Field validation and adaptation.** 494 real images; frozen 0.3361 → 0.6113; adapted 0.8754. Full subtype breakdown showing exactly which contamination types the baseline misses.
4. **Edge deployment.** Jetson Nano CPU-only ONNX, INT8, measured latency/FPS/RSS.

**Headline number:** use the **field** result (0.6113 frozen / 0.8754 adapted vs 0.3361), not the de-leaked 0.9642. Report 0.9642 in a table *next to* the 0.9409 source-only ceiling and state plainly that the in-distribution benchmark is saturated. This costs you an impressive-looking number and buys a defensible paper — and the field result is the more impressive claim anyway.

**Do not** publish 0.9622/0.9642 as a headline achievement. It is ~2 points over a rule that reads no pixels, and a reviewer who notices — or anyone who reproduces it — has the whole paper.

---

## 5. Probability

Target Q3/Q4, must propose a winning model, no new data.

| Plan | P(accept) | Why |
|---|---:|---|
| **Water project, field-robustness framing** | **80–85%** | Every contribution already exists. Work is writing, verification, and honest re-framing — not new science. Genuinely Q2-capable; comfortable at Q3/Q4 |
| Water project, current framing (0.96 headline) | 60–70% | Likely accepted at Q3/Q4, but rests on a number that is ~2 points over no-pixel. Reproduction risk, and a weaker paper |
| aquanet, model-wins framing | 15–25% | No configuration of AquaNet beats its own ablations on the data available. Would require claiming something the tables contradict |
| aquanet, artifact/diagnostic framing | 45–55% | Real finding, but thin alone — better as §4 contribution 1 of the water paper |
| aquanet, original `RESEARCH_PLAN.md` at Q1 | ~5% | Unchanged from `AUDIT.md` |

**Recommended: the first row. ~80–85%.**

The jump from ~10% (where we were two messages ago) to ~80–85% comes entirely from switching which project you write up and which column you report. No new experiments, no new data.

---

## 6. Work plan (~8–10 weeks)

**Phase 1 — Verify the story (1–2 weeks).** Re-derive the source-identity ceiling for every reported benchmark. Confirm the four unverified Kaggle licenses (River Floating Trash, Drone Garbage Detection, Water Pollution Images, Narmada Plastic). Spot-check the auto-assisted Phase 11 field audit labels by hand — currently machine-tagged, and any per-subtype claim rests on them. Check whether `river_floating_trash_rft` train/test units are same-river or same-video, since `source_id` spans splits even though `source_unit` does not.

**Phase 2 — Close the two open gaps (2–3 weeks).** Bootstrap CIs and paired tests on the **field** results, which are now the headline and currently have neither. Manually audit the Phase 8 balanced failure pages (already flagged open in your checklist).

**Phase 3 — Write (3–4 weeks).** Compact `paper_draft_v6` into journal style around the four contributions. Keep `RESEARCH_PLAN.md` §18 evidence rules as the drafting constraint — they are what caught all of this.

**Phase 4 — Venue and submission (1–2 weeks).** Q3/Q4 candidates where this fits: *Journal of Water Process Engineering*, *Water Practice and Technology*, *Environmental Monitoring and Assessment*, *Ecological Informatics* (higher tier — worth one attempt first, the field result may carry it). With the field-robustness framing I would try one Q2 before settling.

**What not to do:** more aquanet training runs; more architecture variants; more seeds on any in-distribution split; the §11 ablation program. None of it moves the outcome.

---

## 7. What to do with aquanet

Three options, in order of preference:

1. **Fold in as the second diagnostic data point.** The metadata-only 68.85% result is a stronger, more visceral demonstration of the ceiling than the water project's own 0.9409, because it uses literally zero pixels. One paragraph and one table in the water paper. Highest value, near-zero cost.
2. **Park it.** `AUDIT.md` documents why. No further compute.
3. **Separate short artifact paper** later, if the water paper lands and there is appetite.

Do not attempt to rescue aquanet as a model-contribution paper. The data cannot support it, and the effort is better spent on the paper that is 80% written.
