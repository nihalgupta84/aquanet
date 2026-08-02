# IrrigWater-7 and AquaNet: Q1 Research and Manuscript Plan

Status: authoritative project roadmap as of July 31, 2026. This document supersedes the leakage-centered framing and the interim four-page `paper_v3` narrative.

> **Rescoped to Q4 — see [`AQUANET_Q4_PLAN.md`](AQUANET_Q4_PLAN.md) (July 31, 2026).** That document is now the authoritative execution plan and supersedes §19. It narrows §7 (drop classical ML and VLM families), §11 and §12 to a Q4 budget, and revises Contribution C1: D1 originates from the public Kaggle dataset `vasundharadixit1826/real-and-ai-data` and contains AI-generated imagery, so it cannot be introduced, named or released as a novel dataset — §5 and C1 are withdrawn and dataset curation moves to methodology. **§18 evidence rules remain in force verbatim.**
>
> Two claims in the earlier audit banner were wrong and are retracted: (a) the ablations do **not** refute C2 — `no_msrb` and `no_csab` were only ever run under the hierarchical head, so MSRB and CSAB have never been tested under the flat head or against a parameter-matched control; (b) the proposed model is **not** losing to the baselines — re-aggregating all 15 completed runs shows AquaNet `no_msrb` beats ResNet50 and MobileNetV2 on test accuracy on all three seeds (0.8774 vs 0.8611 / 0.8603), and under the §8.1 selection rule this plan already specifies (best validation macro-F1), AquaNet `full` ranks first at 0.8761. See `AQUANET_Q4_PLAN.md` §0–§1.

## 1. Paper thesis

The planned paper is a dataset-and-model study for visual irrigation-water condition recognition:

1. Introduce a named, multi-domain, seven-class image dataset.
2. Develop and validate an attention-guided multi-scale recognition architecture through AquaNet v1, v2, v3-H, v3-F, VLM, and controlled variants.
3. Establish a unified benchmark spanning classical ML, CNNs, vision transformers, and vision-language models.
4. Evaluate domain transfer, robustness, calibration, uncertainty, explainability, complexity, and failure modes.

Duplicate control, source separation, reproducible code, and eventual release are quality requirements. They support credibility but are not title-level novelty or standalone contributions.

## 2. Dataset identity

Provisional dataset name: **IrrigWater-7**.

Expanded name: **IrrigWater-7: A Multi-Domain Visual Dataset for Seven-Class Irrigation Water Condition Recognition**.

Classes: clean, algae, debris, foam, oil, turbid, and uncertain.

Published subset names:

- `IW7-Bench`: current D1, 2,799 images; benchmark train/validation/test.
- `IW7-Adapt`: current D2, 213 images; limited real-domain adaptation.
- `IW7-External`: current D3, 146 images; held-out real-domain evaluation.

Before freezing the name, search IEEE Xplore, Scopus, Google Scholar, IEEE DataPort, Zenodo, Kaggle, GitHub, and Hugging Face for collisions.

## 3. Candidate paper titles

Recommended:

> IrrigWater-7: A Multi-Domain Dataset and Attention-Guided Multi-Scale Framework for Visual Irrigation Water Condition Recognition

Measurement-oriented alternative:

> Visual Measurement of Irrigation Water Conditions: The IrrigWater-7 Dataset and an Attention-Guided Multi-Scale Network

Generalization-oriented alternative:

> Multi-Domain Visual Recognition of Irrigation Water Conditions Using IrrigWater-7 and an Attention-Guided Network

AquaNet may remain the model name without appearing in the title.

## 4. Intended contributions

### C1. Named multi-domain dataset

Introduce IrrigWater-7 with benchmark, adaptation, and external-evaluation subsets; document acquisition, class taxonomy, annotation, source composition, limitations, and release conditions.

### C2. Validated model

Develop a model combining dense feature reuse, multi-scale feature extraction, channel-spatial attention, and a classification head selected from validation evidence. The final head may be flat or hierarchical; novelty wording must follow results.

### C3. Cross-family benchmark

Compare handcrafted-feature ML, CNNs, transformers, and VLMs under frozen splits, comparable tuning budgets, shared seeds, and common metrics.

### C4. Domain generalization

Measure `IW7-Bench -> IW7-External` transfer and the benefit/data efficiency of `IW7-Adapt` fine-tuning.

### C5. Model understanding

Support architectural reasoning with repeated ablations, robustness, calibration, genuine saliency/attention analysis, failure cases, complexity, and statistical tests.

## 5. Dataset release package

Prepare before submission even if public release occurs after acceptance:

```text
IrrigWater-7/
├── README.md
├── DATASET_CARD.md
├── LICENSE.md
├── CITATION.cff
├── classes.json
├── manifests/
│   ├── iw7_bench_train.csv
│   ├── iw7_bench_val.csv
│   ├── iw7_bench_test.csv
│   ├── iw7_adapt.csv
│   └── iw7_external.csv
├── source_licenses/
├── annotation_guidelines/
├── audit/
└── checksums/
```

Required manifest fields: image ID, class, subset/split, source type, source URL or device, video/source group, frame/timestamp, natural/generated status, license, annotation method, annotator agreement, MD5, perceptual hash, width, and height.

Proposed availability statement:

> IrrigWater-7, permitted source manifests, preprocessing utilities, configurations, and evaluation code will be released for research use following acceptance. An anonymized package can be provided to editors and reviewers upon request, subject to venue and source-license policies.

## 6. Dataset preparation and analysis

Quality control, not novelty:

- Adjudicate 698 dHash cross-split candidates.
- Recover source/video groups and enforce group-disjoint splits where possible.
- Verify cross-subset independence, corrupt files, labels, licenses, and dimensions.
- Freeze manifests before final tests.

Dataset analysis:

- Class, source, natural/generated, resolution, and aspect-ratio distributions.
- RGB/HSV and texture statistics.
- Inter/intra-class similarity.
- UMAP/t-SNE of frozen embeddings across IW7-Bench/Adapt/External.
- Representative, ambiguous, and failure-prone samples.
- Annotation protocol and agreement.
- Comparison with existing visual water datasets.
- Training-size learning curves.

## 7. Benchmark families

### Classical ML

Features: RGB/HSV histograms, LBP, HOG, GLCM, combined handcrafted features, and optionally frozen ImageNet embeddings.

Models: logistic regression, KNN, Random Forest, RBF-SVM, and XGBoost/LightGBM.

### CNNs

MobileNetV2, EfficientNet-B0, ResNet50, DenseNet121, ConvNeXt-Tiny, ResNeXt50, AquaNet v1, v2, v3-H, v3-F, and the final selected configuration.

### Transformers and VLMs

ViT, DeiT, Swin-Tiny, MaxViT-Tiny, CLIP zero-shot, prompt ensemble, linear probe, fine-tuning, and AquaNet-VLM.

Fairness requirements: frozen splits, consistent resolution where possible, common seed list, comparable training/tuning budget, explicit pretrained weights, shared model-selection rule, and saved per-image probabilities/predictions.

## 8. Final model selection

1. Compare candidate models using validation macro-F1 only.
2. Use accuracy, false-clean rate, NLL, ECE, parameters, and latency as secondary criteria.
3. Do not select from final test rankings.
4. Freeze architecture, loss, augmentation, schedule, checkpoint rule, and seeds.
5. Run final test once per frozen seed.

Current observation, not yet final selection: AquaNet v3-F (DenseNet121 + MSRB + CSAB + flat head) is the best seed-42 Phase-3 run at 88.99% accuracy. Matched seeds are being completed before claiming best overall.

## 9. Final statistical protocol

For the final model and principal baselines:

- At least five independent seeds.
- Mean, sample standard deviation, and 95% confidence intervals.
- Prediction-level bootstrap differences.
- McNemar tests from paired predictions.
- Paired seed-level tests and effect sizes.
- Holm correction across multiple baseline comparisons.

Save per-image IDs, labels, probabilities, predictions, checkpoints, configs, environment metadata, and training histories.

## 10. Metrics

- Accuracy and balanced accuracy.
- Macro/weighted precision, recall, and F1.
- Per-class precision, recall, and F1.
- One-vs-rest AUROC and AUPRC.
- MCC and Cohen's kappa.
- NLL, Brier score, and ECE.
- 95% confidence intervals.
- Binary clean/contaminated sensitivity, specificity, and false-clean rate.
- Contamination-type accuracy and hierarchical consistency where applicable.

## 11. Ablation program

Repeat main ablations over matched seeds:

- Full, no-MSRB, no-CSAB, flat, hard hierarchy, soft hierarchy.
- DenseNet only, DenseNet+MSRB, DenseNet+CSAB, DenseNet+MSRB+CSAB.
- Channel-only and spatial-only attention.
- Residual fusion on/off.
- MSRB dilation and CSAB reduction ratios.
- Random initialization, frozen backbone, partial/full fine-tuning.
- Cross-entropy, weighted CE, focal, weighted focal.
- Sampler only, weighting only, both, neither.
- Flat/hierarchical loss mixing coefficient.
- Augmentation and image-resolution variants.

Report accuracy, macro-F1, calibration, parameters, FLOPs, latency, and deltas with uncertainty.

## 12. Dataset and domain experiments

- Natural-only, generated-only (if applicable), and combined training.
- IW7-Bench to IW7-Adapt and IW7-External.
- IW7-Bench + limited IW7-Adapt to IW7-External.
- 10/25/50/75/100% adaptation-data curves.
- Frozen versus partial versus full fine-tuning.
- Per-source and per-class transfer.
- Domain embedding distances and error correlation.
- Label-noise and training-size sensitivity.

## 13. Robustness and measurement analysis

Test brightness, contrast, blur, Gaussian noise, JPEG compression, resolution, color temperature, glare simulation, and occlusion. Report corruption degradation, calibration, confidence thresholds, selective prediction, accuracy-coverage, and false-clean risk.

For IEEE Transactions on Instrumentation and Measurement, frame the camera/model as a visual measurement system and emphasize acquisition, uncertainty, calibration, repeatability, reliability, and limitations relative to physicochemical sensors.

## 14. Explainability

Use genuine methods:

- Grad-CAM, Grad-CAM++, Score-CAM, and occlusion sensitivity for CNNs.
- CSAB spatial masks and channel-weight distributions.
- MSRB branch activation comparisons.
- Transformer attention rollout and head-wise attention.
- VLM similarity maps.

Add quantitative deletion/insertion or masking tests where feasible. Do not label averaged feature activations as Grad-CAM or claim glare suppression from pictures alone.

## 15. Required figures

Dataset collection pipeline; IW7 subset roles; sample/class/source/resolution distributions; RGB/HSV analysis; embedding UMAP; learning curves; AquaNet evolution; final architecture; MSRB; CSAB; classifier logic; family benchmark; accuracy-parameter-latency Pareto; confusion matrices; per-class F1; ROC/PR; reliability diagram; domain transfer; ablation heatmap; robustness curves; genuine saliency/attention maps; and failure cases.

## 16. Required tables

Literature taxonomy; dataset comparison; IrrigWater-7 composition; training protocol; classical ML; CNN; transformer/VLM; overall ranking; per-class performance; domain generalization; component/loss/head ablations; robustness; calibration; complexity; and statistical comparisons.

## 17. Manuscript structure

1. Introduction
2. Related Work and Taxonomy
3. IrrigWater-7 Dataset
4. Problem Formulation
5. AquaNet Evolution
6. Final Architecture
7. Experimental Protocol
8. Classical ML Benchmark
9. CNN Benchmark
10. Transformer and VLM Benchmark
11. Extensive Ablations
12. Dataset Contribution Analysis
13. Domain Generalization
14. Robustness and Calibration
15. Interpretability
16. Complexity and Deployment Considerations
17. Discussion
18. Limitations
19. Conclusion

## 18. Evidence rules

- No test-driven architecture selection.
- No state-of-the-art claim without comparable external evidence.
- No absolute leakage-free claim.
- No unsupported Grad-CAM, edge, statistical, or causality claims.
- No invented citations, provenance, annotation statistics, or experiments.
- Model superiority requires matched protocols and uncertainty/statistical evidence.

## 19. Execution order and gates

1. Complete current matched AquaNet variant runs.
2. Aggregate and choose candidate from validation evidence.
3. Freeze IrrigWater-7 identity, manifests, provenance, and splits.
4. Adjudicate quality-control candidates.
5. Implement full ML/CNN/transformer/VLM benchmark.
6. Lock final model and run five-seed final evaluation.
7. Complete repeated ablations.
8. Run domain, robustness, calibration, explainability, and complexity studies.
9. Verify literature taxonomy from primary sources.
10. Select target journal based on final contribution.
11. Replace interim paper_v3 with the full evidence-driven manuscript.
12. Compile, audit claims, package dataset/code, and prepare reviewer access.
