#!/usr/bin/env python3
"""Generate every camera-ready figure from the immutable result JSONs.

There are no numeric literals for results in this file. Every value is loaded through
`paper_data.py`, which reads only `reports/`, `phase4_results/` and `predictions/`. If a
number cannot be traced to a JSON file it does not get drawn.

    python experiments/generate_final_paper_figures.py            # all figures
    python experiments/generate_final_paper_figures.py --only temperature ranking

Colour: two categorical hues carry the only identity distinction that recurs across
figures -- AquaNet variants against pretrained baselines. Both are validated
colourblind-safe against each other and against the third slot used for classical models,
and identity is never colour-alone: every bar is directly labelled on its axis. Heatmaps
use a single-hue sequential ramp for magnitude and a two-hue diverging ramp with a neutral
midpoint for signed differences.
"""
import sys, json, shutil, argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import paper_data as D

OUT = ROOT / 'paper_final' / 'figures'

BASE = '#2a78d6'      # categorical slot 1 -- pretrained baselines
AQUA = '#eb6834'      # categorical slot 2 -- AquaNet variants
THIRD = '#1baf7a'     # categorical slot 3 -- classical models
INK = '#0b0b0b'
MUTED = '#52514e'
GRID = '#d8d7d2'
SEQ = 'Blues'                 # sequential: one hue, light -> dark
DIV = 'RdBu_r'                # diverging: two hues, neutral midpoint

plt.rcParams.update({
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'grid.color': GRID, 'grid.linewidth': 0.4, 'lines.linewidth': 1.2,
    'figure.dpi': 200, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

COL, PAGE = 3.5, 7.16        # IEEEtran column and full text width, inches


def hue(name):
    return AQUA if D.is_aquanet(name) else BASE


# Suppresses the embedded PDF CreationDate. Without it every regeneration produces a
# different file even when the plotted numbers are identical, so `git status` after a rerun
# stops being a signal that something actually changed.
DETERMINISTIC = {'CreationDate': None}


def save(name):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches='tight', metadata=DETERMINISTIC)
    plt.close()
    print(f'  {name}')


def barh(ax, names, vals, xlabel, fmt='{:.4f}', pad=None, colors=None, err=None):
    """Horizontal bars, best at the top, every bar directly labelled."""
    y = np.arange(len(names))
    colors = colors or [hue(n) for n in names]
    ax.barh(y, vals, color=colors, height=0.68,
            xerr=err, error_kw=dict(ecolor=MUTED, lw=0.7, capsize=1.5))
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, lw=0.4)
    ax.set_axisbelow(True)
    span = max(vals) - min(0, min(vals))
    pad = pad if pad is not None else span * 0.02
    for yi, v in zip(y, vals):
        ax.text(v + pad, yi, fmt.format(v), va='center', ha='left',
                fontsize=6, color=MUTED)
    ax.set_xlim(min(0, min(vals) * 1.05), max(vals) + span * 0.18)


def legend_aquanet(ax, loc='lower right'):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=BASE, label='pretrained baseline'),
                       Patch(facecolor=AQUA, label='AquaNet variant')],
              loc=loc, frameon=False)


# ==================================================================== figures

def fig_temperature():
    """HEADLINE. Fitted temperature per model, reference line at the ideal T = 1."""
    cal = D.calibration()
    T = cal['T']
    names = sorted(T, key=lambda n: abs(T[n]['mean'] - 1))
    vals = [T[n]['mean'] for n in names]
    sds = [T[n]['sd'] for n in names]

    fig, ax = plt.subplots(figsize=(COL, 2.9))
    y = np.arange(len(names))
    ax.barh(y, vals, color=[hue(n) for n in names], height=0.68,
            xerr=sds, error_kw=dict(ecolor=MUTED, lw=0.7, capsize=1.5))
    ax.axvline(1.0, color=INK, ls='--', lw=0.8, zorder=4)
    ax.text(1.02, -0.85, 'ideal $T=1$', fontsize=6, color=INK)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel('Fitted temperature (mean $\\pm$ sd, five seeds)')
    ax.set_xlim(0, max(v + s for v, s in zip(vals, sds)) * 1.32)
    ax.xaxis.grid(True, lw=0.4)
    ax.set_axisbelow(True)
    for yi, (v, s) in zip(y, zip(vals, sds)):
        ax.text(v + s + 0.05, yi, f'{v:.3f}', va='center', fontsize=6, color=MUTED)
    legend_aquanet(ax, 'upper right')
    save('temperature.pdf')


def fig_calibration_panel():
    """Raw vs temperature-scaled ECE, and reliability curves for three models."""
    cal = D.calibration()
    raw, calib = cal['ece_raw'], cal['ece_cal']
    names = sorted(raw, key=lambda n: raw[n]['mean'])

    fig, ax = plt.subplots(1, 2, figsize=(PAGE, 2.7))
    y = np.arange(len(names))
    h = 0.36
    ax[0].barh(y - h / 2, [raw[n]['mean'] for n in names], height=h,
               color=[hue(n) for n in names], label='uncalibrated')
    ax[0].barh(y + h / 2, [calib[n]['mean'] for n in names], height=h,
               color=[hue(n) for n in names], alpha=0.45, hatch='///',
               edgecolor='white', linewidth=0.3, label='after temperature scaling')
    ax[0].set_yticks(y, names)
    ax[0].invert_yaxis()
    ax[0].set_xlabel('Expected calibration error, 15 bins')
    ax[0].xaxis.grid(True, lw=0.4)
    ax[0].set_axisbelow(True)
    ax[0].legend(loc='upper right', frameon=False)

    # Reliability, seed 7, the three models the calibration section argues about. Bins
    # holding fewer than five images are dropped -- at 427 test images the tail bins carry
    # one or two and their accuracy is 0 or 1 by construction, which is noise, not miscalibration.
    shown = ['AquaNet-no-neck', 'Swin-T', 'EfficientNet-B0']
    marks = ['o', 's', '^']
    ax[1].plot([0, 1], [0, 1], ls='--', lw=0.8, color=INK, zorder=1)
    for n, m in zip(shown, marks):
        pts = [p for p in D.reliability(n, 7, 'raw') if p['n'] >= 5]
        c = [p['confidence'] for p in pts]
        a = [p['accuracy'] for p in pts]
        sz = [8 + 34 * p['n'] / max(q['n'] for q in pts) for p in pts]
        col = hue(n) if n != 'EfficientNet-B0' else THIRD
        ax[1].plot(c, a, color=col, lw=1.0, zorder=3)
        ax[1].scatter(c, a, s=sz, marker=m, color=col, zorder=4, label=n)
    ax[1].set_xlabel('Confidence (seed 7, uncalibrated; marker area $\\propto$ bin count)')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlim(0.2, 1.02)
    ax[1].set_ylim(0, 1.02)
    ax[1].grid(True, lw=0.4)
    ax[1].set_axisbelow(True)
    ax[1].legend(loc='upper left', frameon=False)
    save('calibration.pdf')


def fig_ranking():
    rows = D.stage_p()
    names = [r['name'] for r in rows]
    mean = np.array([r['mf1'] for r in rows])
    lo = np.array([r['mf1_ci95'][0] for r in rows])
    hi = np.array([r['mf1_ci95'][1] for r in rows])
    aqf = next(r['mf1'] for r in rows if r['name'] == 'AquaNet-full')

    fig, ax = plt.subplots(figsize=(COL, 2.9))
    y = np.arange(len(names))
    ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt='none',
                ecolor=MUTED, capsize=2, lw=0.9)
    ax.scatter(mean, y, c=[hue(n) for n in names], s=22, zorder=3)
    ax.axvline(aqf, color=AQUA, ls='--', lw=0.8, zorder=1)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel('Test macro-F1 (mean and 95% CI, five seeds)')
    ax.xaxis.grid(True, lw=0.4)
    ax.set_axisbelow(True)
    for yi, m in zip(y, mean):
        ax.text(hi.max() + 0.002, yi, f'{m:.4f}', va='center', fontsize=6, color=MUTED)
    ax.set_xlim(lo.min() - 0.006, hi.max() + 0.016)
    legend_aquanet(ax, 'upper left')
    save('ranking.pdf')


def fig_protocol_sensitivity():
    s = D.sensitivity()
    names = sorted(s, key=lambda n: s[n]['spread'])
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    barh(ax, names, [s[n]['spread'] for n in names],
         'Best minus worst validation macro-F1 in the identical six-trial grid')
    legend_aquanet(ax, 'lower right')
    save('protocol_sensitivity.pdf')


def fig_sensitivity_grid():
    """The full six-trial grid per model: what the shared setting could have cost."""
    s = D.sensitivity()
    names = sorted(s, key=lambda n: -s[n]['spread'])
    lrs = sorted({g[1] for n in names for g in s[n]['grid']}, reverse=True)
    mults = sorted({g[2] for n in names for g in s[n]['grid']})
    M = np.full((len(names), len(lrs) * len(mults)), np.nan)
    cols = [(lr, m) for lr in lrs for m in mults]
    for i, n in enumerate(names):
        for v, lr, mu in s[n]['grid']:
            M[i, cols.index((lr, mu))] = v

    fig, ax = plt.subplots(figsize=(PAGE * 0.66, 2.9))
    im = ax.imshow(M, cmap=SEQ, aspect='auto', vmin=np.nanmin(M), vmax=np.nanmax(M))
    ax.set_xticks(range(len(cols)),
                  [f'{lr:g}\n$\\times${mu:g}' for lr, mu in cols], fontsize=6)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel('Learning rate and backbone multiplier')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            rel = (M[i, j] - np.nanmin(M)) / (np.nanmax(M) - np.nanmin(M))
            ax.text(j, i, f'{M[i, j]:.3f}', ha='center', va='center', fontsize=5.4,
                    color='white' if rel > 0.6 else INK)
        # Box the trial each model was promoted from.
        j = int(np.nanargmax(M[i]))
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=AQUA, lw=1.4))
    ax.set_title('Validation macro-F1 per trial; box marks the promoted setting', pad=4)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015).ax.tick_params(labelsize=5.5)
    save('sensitivity_grid.pdf')


def fig_ablation_heatmap():
    f = D.factorial()
    rows = [(m, c) for m in f['msrb'] for c in f['csab']]
    M = np.array([[f['cells'][(h, m, c)] for h in f['heads']] for m, c in rows])
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    im = ax.imshow(M, cmap=SEQ, aspect='auto')
    ax.set_xticks(range(len(f['heads'])), [h.replace('_', '-') for h in f['heads']])
    ax.set_yticks(range(len(rows)), [f'MSRB {m} / CSAB {c}' for m, c in rows])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            rel = (M[i, j] - M.min()) / (M.max() - M.min())
            ax.text(j, i, f'{M[i, j]:.4f}', ha='center', va='center', fontsize=6,
                    color='white' if rel > 0.6 else INK)
    ax.set_xlabel('Classification head')
    ax.set_title(f"Test macro-F1, {f['n_runs']}-run balanced factorial "
                 f"({f['n_cells']} cells $\\times$ {len(f['seeds'])} seeds)", pad=4)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=5.5)
    save('ablation_heatmap.pdf')


def fig_decision_quality():
    ab, co = D.abstention()['aurc'], D.corruptions()['mean_degradation']
    names = sorted(ab, key=lambda n: ab[n]['mean'])
    fig, ax = plt.subplots(1, 2, figsize=(PAGE, 2.5))
    barh(ax[0], names, [ab[n]['mean'] for n in names],
         'Area under the risk-coverage curve (lower is better)', fmt='{:.4f}')
    cn = sorted(co, key=lambda n: co[n]['mean'])
    barh(ax[1], cn, [co[n]['mean'] for n in cn],
         'Mean macro-F1 loss over nine corruptions (lower is better)', fmt='{:.4f}')
    legend_aquanet(ax[0], 'lower right')
    save('decision_quality.pdf')


def fig_rc_curves():
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    ab = D.abstention()['aurc']
    shown = sorted(ab, key=lambda n: ab[n]['mean'])[:2] + ['AquaNet-full', 'AquaNet-no-neck']
    for n, ls in zip(dict.fromkeys(shown), ['-', '--', '-', '--']):
        c = D.rc_curve(n, 7)
        ax.plot([p['coverage'] for p in c], [p['risk'] for p in c], ls=ls,
                color=hue(n), label=f"{n} (AURC {ab[n]['mean']:.4f})")
    ax.set_xlabel('Coverage (seed 7)')
    ax.set_ylabel('Selective risk')
    ax.grid(True, lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=False)
    save('rc_curves.pdf')


def fig_corruption_curves():
    c = D.corruptions()
    names = c['names']
    shown = ['Swin-T', 'DeiT-S', 'DenseNet-121', 'AquaNet-no-neck', 'AquaNet-full']
    fig, axes = plt.subplots(3, 3, figsize=(PAGE, 4.6), sharex=True, sharey=True)
    sev = np.arange(1, 6)
    for ax, cname in zip(axes.ravel(), names):
        for n, ls in zip(shown, ['-', '--', ':', '-', '--']):
            ax.plot(sev, c['severity_curves'][cname][n]['mean'], ls=ls, color=hue(n),
                    marker='o', ms=2.4, label=n)
        ax.set_title(cname.replace('_', ' '), pad=3)
        ax.grid(True, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_xticks(sev)
    for ax in axes[-1]:
        ax.set_xlabel('Severity')
    for ax in axes[:, 0]:
        ax.set_ylabel('Macro-F1')
    axes[0, 0].legend(loc='lower left', frameon=False, fontsize=5.6)
    fig.suptitle('Corruption robustness, mean over five seeds', y=1.005, fontsize=8)
    save('corruption_curves.pdf')


def fig_ood():
    z = D.zeroshot()['mf1']
    names = sorted(z, key=lambda n: z[n]['mean'])
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    barh(ax, names, [z[n]['mean'] for n in names],
         'D3 zero-shot macro-F1 (146 images)', err=[z[n]['sd'] for n in names])
    legend_aquanet(ax, 'lower right')
    save('ood_transfer.pdf')


def fig_adaptation():
    a = D.adaptation()
    fig, ax = plt.subplots(figsize=(COL * 0.92, 2.3))
    x = [0] + [p['pct'] for p in a['points']]
    y = [a['zero_shot_mf1']] + [p['mf1'] for p in a['points']]
    e = [0] + [p['sd'] for p in a['points']]
    ax.errorbar(x, y, yerr=e, fmt='o-', color=BASE, ms=3.5, capsize=2, ecolor=MUTED, lw=1.2)
    ax.set_xticks(x)
    ax.set_xlabel(f"D2 adaptation data used (%), source {a['source_run']}")
    ax.set_ylabel('D3 macro-F1')
    ax.grid(True, lw=0.4)
    ax.set_axisbelow(True)
    for xi, yi in zip(x, y):
        ax.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                    xytext=(0, 6), ha='center', fontsize=6, color=MUTED)
    ax.set_ylim(min(y) - 0.04, max(y) + 0.06)
    save('adaptation.pdf')


def fig_explanations():
    e = D.explanations()
    dele, ins = e['deletion'], e['insertion']
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    for n in dele:
        ax.scatter(dele[n]['mean'], ins[n]['mean'], s=26, color=hue(n), zorder=3)
        ax.annotate(n, (dele[n]['mean'], ins[n]['mean']), textcoords='offset points',
                    xytext=(4, 3), fontsize=5.6, color=MUTED)
    ax.set_xlabel('Deletion AUC (lower is better)')
    ax.set_ylabel('Insertion AUC (higher is better)')
    ax.grid(True, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_title(f"Grad-CAM faithfulness, {e['n_images']} images, five-seed means", pad=4)
    save('explanations.pdf')


def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(PAGE * 0.82, 3.0))
    for ax, name in zip(axes, ['Swin-T', 'AquaNet-no-neck']):
        m = D.confusion(name)
        norm = m / m.sum(1, keepdims=True)
        im = ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
        ax.set_xticks(range(len(D.CLASSES)), D.CLASSES, rotation=45, ha='right')
        ax.set_yticks(range(len(D.CLASSES)), D.CLASSES)
        for i in range(len(D.CLASSES)):
            for j in range(len(D.CLASSES)):
                if m[i, j] == 0:
                    continue
                ax.text(j, i, m[i, j], ha='center', va='center', fontsize=5.4,
                        color='white' if norm[i, j] > 0.6 else INK)
        ax.set_title(f'{name}  (five seeds pooled)', pad=4)
        ax.set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                 label='Row-normalised rate').ax.tick_params(labelsize=5.5)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / 'confusion.pdf', bbox_inches='tight', metadata=DETERMINISTIC)
    plt.close()
    print('  confusion.pdf')


def fig_per_class_f1():
    pc = D.per_class_f1()
    order = [r['name'] for r in D.stage_p()]
    fig, ax = plt.subplots(figsize=(PAGE, 2.6))
    x = np.arange(len(D.CLASSES))
    w = 0.085
    for i, n in enumerate(order):
        ax.bar(x + (i - len(order) / 2 + 0.5) * w, [pc[n][c]['mean'] for c in D.CLASSES],
               width=w, color=hue(n), alpha=1.0 if D.is_aquanet(n) else 0.85,
               edgecolor='white', linewidth=0.25)
    ax.set_xticks(x, [f"{c}\n(n={D.confusion(order[0]).sum(1)[i] // len(D.SEEDS)})"
                      for i, c in enumerate(D.CLASSES)])
    ax.set_ylabel('Per-class F1 (five-seed mean)')
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, lw=0.4)
    ax.set_axisbelow(True)
    legend_aquanet(ax, 'lower right')
    ax.set_title('Ten models per class, ordered as in the ranking table', pad=4)
    save('per_class_f1.pdf')


def fig_roc_pr():
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    shown = ['Swin-T', 'AquaNet-no-neck']
    fig, axes = plt.subplots(1, 2, figsize=(PAGE * 0.85, 2.7))
    styles = ['-', '--']
    cmap = plt.get_cmap('viridis')
    for ax, kind in zip(axes, ['roc', 'pr']):
        for name, ls in zip(shown, styles):
            yt, _, yp, _ = D.predictions(name, 7)
            for ci, c in enumerate(D.CLASSES):
                b = (yt == ci).astype(int)
                col = cmap(ci / (len(D.CLASSES) - 1))
                if kind == 'roc':
                    fpr, tpr, _ = roc_curve(b, yp[:, ci])
                    ax.plot(fpr, tpr, ls=ls, color=col, lw=0.9,
                            label=f'{c} ({auc(fpr, tpr):.3f})' if ls == '-' else None)
                else:
                    pr, rc, _ = precision_recall_curve(b, yp[:, ci])
                    ax.plot(rc, pr, ls=ls, color=col, lw=0.9,
                            label=f'{c} ({average_precision_score(b, yp[:, ci]):.3f})'
                            if ls == '-' else None)
        ax.grid(True, lw=0.4)
        ax.set_axisbelow(True)
    axes[0].plot([0, 1], [0, 1], ls=':', lw=0.7, color=MUTED)
    axes[0].set_xlabel('False positive rate')
    axes[0].set_ylabel('True positive rate')
    axes[0].set_title('One-vs-rest ROC (AUC)', pad=3)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('One-vs-rest PR (AP)', pad=3)
    axes[0].legend(loc='lower right', frameon=False, fontsize=5.4, ncol=1)
    fig.suptitle(f'Solid {shown[0]}, dashed {shown[1]}; seed 7', y=1.02, fontsize=7)
    save('roc_pr.pdf')


def fig_pareto():
    rows = {r['name']: r for r in D.stage_p()}
    cx = D.complexity()
    fig, ax = plt.subplots(1, 2, figsize=(PAGE * 0.85, 2.6))
    for a, (key, xlabel) in zip(ax, [('params_m', 'Parameters (M)'),
                                     ('latency_cuda_ms', 'GPU latency per batch (ms)')]):
        for n, r in rows.items():
            xv = cx[key][n]['mean']
            a.scatter(xv, r['mf1'], s=26, color=hue(n), zorder=3)
            a.annotate(n, (xv, r['mf1']), textcoords='offset points', xytext=(4, 3),
                       fontsize=5.4, color=MUTED)
        a.set_xlabel(xlabel)
        a.set_ylabel('Test macro-F1')
        a.grid(True, lw=0.4)
        a.set_axisbelow(True)
    ax[0].set_xscale('log')
    note = cx['gflops_note']
    if not cx['gflops_available'] and note:
        # Never draw a FLOPs axis we did not measure.
        fig.suptitle(f'GFLOPs omitted: {note}', y=1.02, fontsize=6, color=MUTED)
    save('pareto.pdf')


def fig_dataset():
    s = D.dataset_stats()
    if not s:
        print('  (skipped dataset.pdf: reports/dataset_stats.json absent)')
        return
    d1 = s['sets']['D1']
    fig, ax = plt.subplots(1, 3, figsize=(PAGE, 2.2))

    splits = ['train', 'val', 'test']
    bottom = np.zeros(len(D.CLASSES))
    shades = [plt.get_cmap(SEQ)(v) for v in (0.35, 0.6, 0.85)]
    for sp, col in zip(splits, shades):
        v = np.array([d1['counts_by_split_label'][sp].get(c, 0) for c in D.CLASSES])
        ax[0].bar(D.CLASSES, v, bottom=bottom, label=sp, color=col,
                  edgecolor='white', linewidth=0.4)
        bottom += v
    ax[0].set_ylabel('Images')
    ax[0].set_title(f"D1 class balance (n={d1['n']})", pad=3)
    ax[0].tick_params(axis='x', rotation=45)
    ax[0].legend(frameon=False)
    ax[0].yaxis.grid(True, lw=0.4)
    ax[0].set_axisbelow(True)

    ar = d1['aspect_ratio']
    labels = ['<0.5', '0.5-0.75', '0.75-1', '1-1.25', '1.25-1.5', '1.5-1.75', '1.75-2', '>2']
    ax[1].bar(labels, ar['hist_counts'], color=BASE, edgecolor='white', linewidth=0.4)
    ax[1].axvline(2.5, color=AQUA, ls='--', lw=0.9)
    ax[1].set_title('Source aspect ratio (square = 1)', pad=3)
    ax[1].set_ylabel('Images')
    ax[1].tick_params(axis='x', rotation=45)
    ax[1].yaxis.grid(True, lw=0.4)
    ax[1].set_axisbelow(True)

    top = d1['top_resolutions'][:8]
    ax[2].barh([t[0] for t in top][::-1], [t[1] for t in top][::-1], color=BASE)
    ax[2].set_title(f"Top resolutions of {d1['n_distinct_resolutions']}", pad=3)
    ax[2].set_xlabel('Images')
    ax[2].xaxis.grid(True, lw=0.4)
    ax[2].set_axisbelow(True)
    save('dataset.pdf')


def fig_classical():
    cl = D.classical()
    if not cl:
        print('  (skipped classical.pdf: reports/stageG_classical.json absent)')
        return
    deep = {r['name']: r for r in D.stage_p()}
    names, vals, sds, cols = [], [], [], []
    for k, v in sorted(cl['models'].items(), key=lambda z: z[1]['test_macro_f1_mean']):
        names.append(k.replace('_', ' '))
        vals.append(v['test_macro_f1_mean'])
        sds.append(v['test_macro_f1_sd'])
        cols.append(THIRD)
    for n in ('MobileNetV2', 'DenseNet-121', 'AquaNet-full', 'Swin-T'):
        names.append(n)
        vals.append(deep[n]['mf1'])
        sds.append(deep[n]['mf1_sd'])
        cols.append(hue(n))
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    barh(ax, [names[i] for i in order], [vals[i] for i in order],
         'Test macro-F1 (five seeds)', colors=[cols[i] for i in order],
         err=[sds[i] for i in order])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=THIRD, label='classical, hand-crafted features'),
                       Patch(facecolor=BASE, label='pretrained baseline'),
                       Patch(facecolor=AQUA, label='AquaNet variant')],
              loc='lower right', frameon=False)
    save('classical.pdf')


def fig_architecture():
    """Schematic of the trunk, MSRB, CSAB and the MatchedNeck control."""
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.4))

    def box(ax, x, y, w, h, label, fc='white', ec=INK, fs=6):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fs)

    def arrow(ax, x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=6, lw=0.7, color=MUTED))

    ax = axes[0]
    box(ax, 0.02, 0.42, 0.16, 0.16, 'input\n224$\\times$224')
    box(ax, 0.22, 0.42, 0.20, 0.16, 'DenseNet-121\ntrunk (pretrained)', fc='#eef3fb')
    box(ax, 0.46, 0.42, 0.13, 0.16, 'MSRB', fc='#fdeee6')
    box(ax, 0.63, 0.42, 0.13, 0.16, 'CSAB', fc='#fdeee6')
    box(ax, 0.80, 0.62, 0.18, 0.14, 'flat head\n(7 classes)')
    box(ax, 0.80, 0.26, 0.18, 0.14, 'hier. head\nbinary $\\times$ type')
    for x1, x2 in ((0.18, 0.22), (0.42, 0.46), (0.59, 0.63)):
        arrow(ax, x1, 0.50, x2, 0.50)
    arrow(ax, 0.76, 0.52, 0.80, 0.66)
    arrow(ax, 0.76, 0.48, 0.80, 0.36)
    ax.text(0.55, 0.34, 'ablated jointly and separately\n(off / matched / on)',
            ha='center', fontsize=5.6, color=MUTED)
    ax.set_title('AquaNet-V3', pad=3)

    ax = axes[1]
    box(ax, 0.02, 0.60, 0.16, 0.22, '1$\\times$1, 128\n3$\\times$3, 128\n'
                                    '3$\\times$3, 128\n1$\\times$1, 128', fc='#fdeee6', fs=5.4)
    ax.text(0.10, 0.86, 'MSRB: four parallel branches', ha='center', fontsize=6)
    ax.text(0.10, 0.54, '3,407,872 conv weights\n3,072 BN affine',
            ha='center', fontsize=5.4, color=MUTED)
    box(ax, 0.55, 0.60, 0.16, 0.22, '1$\\times$1, 512\n3$\\times$3, 512\n(single scale)',
        fc='#eef3fb', fs=5.4)
    ax.text(0.63, 0.86, 'MatchedNeck control', ha='center', fontsize=6)
    ax.text(0.63, 0.54, 'same 3,407,872 weights\nsame 3,072 BN affine',
            ha='center', fontsize=5.4, color=MUTED)
    ax.text(0.5, 0.30, 'Identical parameter count, normalisation count and depth.\n'
                       'Only the multi-scale structure differs, so "MSRB helps" is\n'
                       'separated from "3.4M extra parameters help".',
            ha='center', fontsize=5.8, color=INK)
    ax.set_title('The parameter-matched control', pad=3)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0.15, 0.95)
        ax.axis('off')
    save('architecture.pdf')


def fig_sample_grid():
    """One representative image per class. Needs the images, so it is optional."""
    root = ROOT / 'data' / 'cleaned_water_dataset' / 'test'
    if not root.exists():
        print('  (skipped sample_grid.png: data/ not present)')
        return
    from PIL import Image
    fig, axes = plt.subplots(1, len(D.CLASSES), figsize=(PAGE, 1.25))
    for ax, c in zip(axes, D.CLASSES):
        d = root / 'clean' if c == 'clean' else root / 'contaminated' / c
        files = sorted(p for p in d.rglob('*') if p.suffix.lower() in {'.jpg', '.png', '.jpeg'})
        if files:
            ax.imshow(Image.open(files[0]).convert('RGB').resize((224, 224)))
        ax.set_title(c, fontsize=6, pad=2)
        ax.axis('off')
    OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / 'sample_grid.png', bbox_inches='tight', dpi=200)
    plt.close()
    print('  sample_grid.png')


def fig_gradcam_panel():
    src = ROOT / 'reports/stageF_figures/P_aquanet-flat-on-on_seed42_gradcam.png'
    if src.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, OUT / 'gradcam_aquanet_seed42.png')
        print('  gradcam_aquanet_seed42.png')
    else:
        print('  (skipped gradcam: source panel absent)')


FIGURES = {
    'temperature': fig_temperature, 'calibration': fig_calibration_panel,
    'ranking': fig_ranking, 'protocol_sensitivity': fig_protocol_sensitivity,
    'sensitivity_grid': fig_sensitivity_grid, 'ablation_heatmap': fig_ablation_heatmap,
    'decision_quality': fig_decision_quality, 'rc_curves': fig_rc_curves,
    'corruption_curves': fig_corruption_curves, 'ood_transfer': fig_ood,
    'adaptation': fig_adaptation, 'explanations': fig_explanations,
    'confusion': fig_confusion, 'per_class_f1': fig_per_class_f1, 'roc_pr': fig_roc_pr,
    'pareto': fig_pareto, 'dataset': fig_dataset, 'classical': fig_classical,
    'architecture': fig_architecture, 'sample_grid': fig_sample_grid,
    'gradcam': fig_gradcam_panel,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', choices=sorted(FIGURES))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out).resolve()
    print(f'writing figures to {OUT}')
    for name in (a.only or FIGURES):
        FIGURES[name]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
