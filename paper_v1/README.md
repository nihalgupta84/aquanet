# IEEE Journal LaTeX Template — AquaNet / Irrigation Water Quality Monitor

This folder contains a ready-to-edit IEEE journal manuscript skeleton for the AquaNet canal-camera water contamination project.

## Files

- `main.tex` — main IEEE journal LaTeX manuscript.
- `references.bib` — starter BibTeX file with TODO entries.
- `figures/` — put final pipeline, architecture, dataset, graphs, and qualitative images here.

## How to use in Overleaf

1. Upload this ZIP to Overleaf.
2. Set `main.tex` as the main file.
3. Replace all `[TODO: ...]` fields.
4. Add your actual figures in `figures/`.
5. Replace placeholder figure boxes with `\includegraphics`.
6. Add verified bibliographic entries in `references.bib`.
7. Compile with pdfLaTeX + BibTeX.

## Figure placeholders to prepare

- Main pipeline figure.
- AquaNet full architecture.
- MSRB internal block.
- CSAB internal block.
- Three workflow blocks:
  - dataset preparation/training workflow,
  - web application workflow,
  - IoT canal-camera workflow.
- Dataset distribution bar chart.
- Dataset distribution pie chart.
- Dataset grid image.
- Qualitative results grid.
- Confusion matrices.

## Tables already included

- Paper taxonomy.
- Implementation details.
- Dataset summary.
- Class-wise dataset distribution.
- Baselines on synthetic and real test sets.
- Synthetic-training transfer evaluation.
- Real-data fine-tuning evaluation.
- Model complexity.
- MSRB ablation.
- CSAB ablation.
- Combined block ablation.
- Binary real-set classification.
- Multi-class real-set classification.

## Notes

The template uses `\documentclass[journal]{IEEEtran}`. For the exact target IEEE journal, always confirm the latest author guidelines and template settings using the IEEE Template Selector before submission.
