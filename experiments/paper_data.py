"""Single source of truth for every number that reaches the manuscript.

Every figure and every table is built from this module, and this module reads only the
immutable result JSONs -- nothing else, and nothing computed on the fly from images or
checkpoints. Nothing here returns a literal. If a number is not in a JSON file it does not
appear in the paper.

Naming: three schemes coexist in the artefacts and are normalised to one pretty label here.

    phase4_results / reports tag   P_aquanet-flat-on-on_seed7   P_swin_tiny_seed7
    final_report config key        aquanet[flat,msrb=on,csab=on]
    pretty label                   AquaNet-full                 Swin-T
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# This module is shared verbatim between the working repository and the public release,
# which lay the same artefacts out differently. Each logical result is therefore resolved
# by trying both locations rather than hardcoding either.
#
#   working repo                      public release
#   reports/stageD_calibration.json   results/calibration.json
#   reports/final/final_report.json   results/final_report.json
#   phase4_results/                   results/runs/
#   predictions/phase4/               predictions/
_LAYOUTS = [
    {'reports': ROOT / 'reports', 'runs': ROOT / 'phase4_results',
     'preds': ROOT / 'predictions' / 'phase4', 'prefix': True},
    {'reports': ROOT / 'results', 'runs': ROOT / 'results' / 'runs',
     'preds': ROOT / 'predictions', 'prefix': False},
]
_LAYOUT = next((l for l in _LAYOUTS if l['runs'].is_dir()), _LAYOUTS[0])

REPORTS = _LAYOUT['reports']
P4 = _LAYOUT['runs']
PRED4 = _LAYOUT['preds']

# logical name -> filename in the working repo (stage-prefixed) and in the release
_FILES = {
    'final_report': ('final/final_report.json', 'final_report.json'),
    'calibration': ('stageD_calibration.json', 'calibration.json'),
    'abstention': ('stageD_abstention.json', 'abstention.json'),
    'binary': ('stageD_binary.json', 'binary.json'),
    'corruptions': ('stageD_corruptions.json', 'corruptions.json'),
    'complexity': ('stageD_complexity.json', 'complexity.json'),
    'zeroshot': ('stageE_zeroshot.json', 'zeroshot.json'),
    'deletion': ('stageF_deletion.json', 'deletion.json'),
    'gradcam': ('stageF_gradcam.json', 'gradcam.json'),
    'classical': ('stageG_classical.json', 'classical.json'),
    'shortcut': ('stageH_shortcut.json', 'shortcut.json'),
    'dataset_stats': ('dataset_stats.json', 'dataset_stats.json'),
    'dataset_audit': ('../phase3_results/dataset_audit.json', 'dataset_audit.json'),
}


def path_for(name):
    """Absolute path of a logical result file under whichever layout is present."""
    return (REPORTS / _FILES[name][0 if _LAYOUT['prefix'] else 1]).resolve()


def adapt_path(pct, seed):
    stem = (f'stageE_adapt_{pct}pct_seed{seed}.json' if _LAYOUT['prefix']
            else f'adapt_{pct}pct_seed{seed}.json')
    return REPORTS / stem

SEEDS = [7, 21, 42, 1337, 2024]
CLASSES = ['clean', 'algae', 'debris', 'foam', 'oil', 'turbid', 'uncertain']

# tag stem -> pretty label
TAG2PRETTY = {
    'swin_tiny': 'Swin-T', 'deit_small': 'DeiT-S', 'convnext_tiny': 'ConvNeXt-T',
    'resnet50': 'ResNet-50', 'densenet121': 'DenseNet-121',
    'efficientnet_b0': 'EfficientNet-B0', 'mobilenetv2': 'MobileNetV2',
    'aquanet-flat-off-off': 'AquaNet-no-neck', 'aquanet-flat-on-on': 'AquaNet-full',
    'aquanet-hier_tf-on-off': 'AquaNet-hier',
}
PRETTY2TAG = {v: k for k, v in TAG2PRETTY.items()}
# final_report.json config key -> pretty label
CFG2PRETTY = dict(TAG2PRETTY)
CFG2PRETTY.update({
    'aquanet[flat,msrb=off,csab=off]': 'AquaNet-no-neck',
    'aquanet[flat,msrb=on,csab=on]': 'AquaNet-full',
    'aquanet[hier_tf,msrb=on,csab=off]': 'AquaNet-hier',
})
AQUANET = ('AquaNet-full', 'AquaNet-no-neck', 'AquaNet-hier')


def is_aquanet(name):
    return name in AQUANET


def _load(name):
    return json.loads(path_for(name).read_text())


def _any_entry(d):
    """First per-run record of a stage report, for fields identical across runs."""
    for k, v in d.items():
        if not k.startswith('_'):
            return v
    raise ValueError('stage report contains no run entries')


def _first_entry(name, keys):
    v = _any_entry(_load(name))
    return {k: v[k] for k in keys}


def _stem(tag):
    """'P_aquanet-flat-on-on_seed7' -> 'aquanet-flat-on-on'"""
    return tag.split('_', 1)[1].rsplit('_seed', 1)[0]


def _agg(name, fn):
    """{pretty: {'mean','sd','n','by_seed'}} over the per-run entries of a stage report."""
    d = _load(name)
    g = defaultdict(dict)
    for tag, v in d.items():
        if tag.startswith('_'):
            continue
        g[TAG2PRETTY.get(_stem(tag), _stem(tag))][int(tag.rsplit('_seed', 1)[1])] = fn(v)
    out = {}
    for k, by_seed in g.items():
        vals = [by_seed[s] for s in sorted(by_seed)]
        out[k] = {'mean': float(np.mean(vals)),
                  'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                  'n': len(vals), 'by_seed': by_seed}
    return out


# ------------------------------------------------------------------ Stage P headline

def stage_p():
    """Five-seed finalists, ranked by test macro-F1. Selection was on validation macro-F1."""
    rep = _load('final_report')
    rows = []
    for s in rep['summary']:
        rows.append({
            'name': CFG2PRETTY.get(s['config'], s['config']), 'config': s['config'],
            'n_seeds': s['n_seeds'], 'seeds': s['seeds'],
            'val_mf1': s['val_macro_f1_mean'],
            'acc': s['test_acc_mean'], 'acc_sd': s['test_acc_sd'],
            'mf1': s['test_macro_f1_mean'], 'mf1_sd': s['test_macro_f1_sd'],
            'mf1_ci95': s['test_macro_f1_ci95'], 'acc_ci95': s['test_acc_ci95'],
            'params_m': s['params_m'],
        })
    return sorted(rows, key=lambda r: -r['mf1'])


def selected_model():
    return CFG2PRETTY[_load('final_report')['selected']]


def paired_tests(metric=None):
    rep = _load('final_report')
    out = [{**c, 'name': CFG2PRETTY.get(c['vs'], c['vs'])} for c in rep['paired_seed_tests']]
    return [c for c in out if metric is None or c['metric'] == metric]


def prediction_level_tests():
    rep = _load('final_report')
    return [{**t, 'name': CFG2PRETTY.get(t['vs'], t['vs'])}
            for t in rep.get('prediction_level_tests', [])]


# ------------------------------------------------------------------ Stage D / E / F

def calibration():
    """Temperature, ECE, NLL and Brier, raw and after temperature scaling."""
    f = 'calibration'
    return {
        'T': _agg(f, lambda v: v['temperature_fitted_on_val']),
        'ece_raw': _agg(f, lambda v: v['ece_raw']),
        'ece_cal': _agg(f, lambda v: v['ece_calibrated']),
        'nll_raw': _agg(f, lambda v: v['nll_raw']),
        'nll_cal': _agg(f, lambda v: v['nll_calibrated']),
        'brier_raw': _agg(f, lambda v: v['brier_raw']),
        'brier_cal': _agg(f, lambda v: v['brier_calibrated']),
    }


def reliability(pretty, seed, which='raw'):
    d = _load('calibration')
    return d[f'P_{PRETTY2TAG[pretty]}_seed{seed}'][f'reliability_{which}']


def abstention():
    f = 'abstention'
    return {'aurc': _agg(f, lambda v: v['aurc']),
            'acc_cov80': _agg(f, lambda v: v['acc@cov80']),
            'acc_cov90': _agg(f, lambda v: v['acc@cov90']),
            'acc_cov95': _agg(f, lambda v: v['acc@cov95'])}


def rc_curve(pretty, seed):
    d = _load('abstention')
    return d[f'P_{PRETTY2TAG[pretty]}_seed{seed}']['curve']


def binary_screening():
    f = 'binary'
    return {'false_clean': _agg(f, lambda v: v['false_clean_rate']),
            'auroc': _agg(f, lambda v: v['auroc']),
            'sensitivity': _agg(f, lambda v: v['sensitivity_contaminated']),
            'specificity': _agg(f, lambda v: v['specificity_clean']),
            **_first_entry(f, ('n_clean', 'n_contaminated'))}


def corruption_names():
    return list(_any_entry(_load('corruptions'))['corruptions'].keys())


def corruptions():
    """Mean degradation overall, plus the per-corruption severity curves."""
    f = 'corruptions'
    names = corruption_names()
    out = {'mean_degradation': _agg(f, lambda v: float(np.mean(
        [c['degradation'] for c in v['corruptions'].values()]))),
        'clean_mf1': _agg(f, lambda v: v['clean_macro_f1']),
        'names': names, 'by_corruption': {}, 'severity_curves': {}}
    for c in names:
        out['by_corruption'][c] = _agg(f, lambda v, c=c: v['corruptions'][c]['degradation'])
        out['severity_curves'][c] = _agg(
            f, lambda v, c=c: v['corruptions'][c]['macro_f1_by_severity'])
        # _agg means over seeds elementwise for the 5-severity vector
        for k, e in out['severity_curves'][c].items():
            arr = np.array([e['by_seed'][s] for s in sorted(e['by_seed'])])
            e['mean'] = arr.mean(0).tolist()
            e['sd'] = arr.std(0, ddof=1).tolist()
    return out


def zeroshot():
    f = 'zeroshot'
    out = {'mf1': _agg(f, lambda v: v['macro_f1']),
           'acc': _agg(f, lambda v: v['accuracy']),
           'false_clean': _agg(f, lambda v: v['binary_false_clean_rate'])}
    d = _load(f)
    out['per_class_n'] = {c: v['n'] for c, v in _any_entry(d)['per_class'].items()}
    out['per_class_recall'] = {
        c: _agg(f, lambda v, c=c: v['per_class'][c]['recall']) for c in CLASSES}
    return out


def adaptation(seeds=(7, 21, 42)):
    """D2 fine-tuning curve. `before` is the same zero-shot point at every fraction."""
    pts = []
    src = None
    for pct in (25, 50, 75, 100):
        after, before, n_img = [], [], None
        for s in seeds:
            d = json.loads(adapt_path(pct, s).read_text())
            after.append(d['after']['macro_f1'])
            before.append(d['before']['macro_f1'])
            n_img, src = d['n_adaptation_images'], d['source_run']
        pts.append({'pct': pct, 'n_images': n_img,
                    'mf1': float(np.mean(after)), 'sd': float(np.std(after, ddof=1)),
                    'n_seeds': len(after)})
    zero = float(np.mean(before))
    return {'source_run': src, 'zero_shot_mf1': zero, 'points': pts, 'seeds': list(seeds)}


def explanations():
    f = 'deletion'
    return {'deletion': _agg(f, lambda v: v['deletion_auc']),
            'insertion': _agg(f, lambda v: v['insertion_auc']),
            'n_images': _any_entry(_load(f))['n_images']}


def complexity():
    f = 'complexity'
    d = _load(f)
    out = {'params_m': _agg(f, lambda v: v['params_m']),
           'latency_cuda_ms': _agg(f, lambda v: v['latency_ms_cuda']),
           'latency_cpu_ms': _agg(f, lambda v: v['latency_ms_cpu']),
           'peak_mem_mib': _agg(f, lambda v: v['peak_mem_mib'])}
    # GFLOPs are reported only if fvcore actually measured them. Never estimated.
    gf = [v.get('gflops') for k, v in d.items() if not k.startswith('_')]
    out['gflops_available'] = any(g is not None for g in gf)
    out['gflops_note'] = next((v.get('gflops_note') for k, v in d.items()
                               if not k.startswith('_') and v.get('gflops_note')), None)
    return out


# ------------------------------------------------------------------ Stage T / B

def sensitivity():
    """Best-minus-worst validation macro-F1 within the identical six-trial grid.

    Replicates select_lr.py's selection so the figure and the appendix table cannot drift.
    """
    rows = defaultdict(list)
    for f in sorted(P4.glob('T_*.json')):
        r = json.loads(f.read_text())
        c = r.get('config', {})
        if c.get('lr') is None or c.get('backbone_lr_mult') is None:
            continue
        if 'best_val_macro_f1' not in r:
            continue
        key = (r['model'] if r['model'] != 'aquanet_v3'
               else f"aquanet-{r['head']}-{r['msrb']}-{r['csab']}")
        rows[key].append((r['best_val_macro_f1'], c['lr'], c['backbone_lr_mult']))
    out = {}
    for k, rs in rows.items():
        hi = max(rs)
        out[TAG2PRETTY.get(k, k)] = {
            'n_trials': len(rs), 'best_val_mf1': hi[0], 'worst_val_mf1': min(r[0] for r in rs),
            'spread': hi[0] - min(r[0] for r in rs), 'best_lr': hi[1], 'best_mult': hi[2],
            'grid': sorted(rs, reverse=True)}
    return out


def _stage_b_runs():
    """Stage B split into the balanced factorial and the auxiliary sensitivity runs.

    69 files carry `stage == 'B'`, but they are not 69 factorial replicates. 54 of them form
    the balanced 3 heads x 3 MSRB x 2 CSAB x 3 seeds design at the base configuration. The
    remaining 15 are one-off sensitivity runs that exist for the (hier_tf, on, on) cell only:
    a lambda_mix sweep and an uncertain_binary=exclude variant. Averaging them into the
    marginals lets a single cell's side experiments move an axis it should not touch, so
    they are separated here and reported on their own.
    """
    runs = [json.loads(f.read_text()) for f in sorted(P4.glob('B_*.json'))]
    runs = [r for r in runs if 'macro_f1' in r]
    base, aux = [], []
    for r in runs:
        c = r.get('config', {})
        (base if c.get('lambda_mix', 0.5) == 0.5 and c.get('uncertain_binary') == 'contam'
         else aux).append(r)
    return base, aux


def shared_setting():
    """The shared-hyperparameter arm: what every model scored at one common setting.

    Stages A and C both ran at lr = 3e-4, backbone multiplier 0.1 -- the single setting the
    original study used for every model. They overlap: for MobileNetV2 and ResNet-50 the
    same seeds appear under both stage tags, and for all but one of those pairs the stored
    macro-F1 is bit-identical, i.e. Stage C re-tagged the Stage A run. Runs are therefore
    deduplicated on (config, seed) with Stage A taking precedence, so no seed is counted
    twice. ResNet-50 seed 7 is the one genuine disagreement (A .8247 vs C .7869); Stage A
    is used, which is the *conservative* choice because it makes ResNet-50's gain from
    tuning smaller, not larger.

    Returns {pretty: {'shared_mf1', 'tuned_mf1', 'gain', 'n_matched', 'seeds'}} where the
    gain is computed on seeds present in both arms only.
    """
    by = defaultdict(dict)
    for stage in ('A', 'C'):
        for f in sorted(P4.glob(f'{stage}_*.json')):
            r = json.loads(f.read_text())
            if 'macro_f1' not in r:
                continue
            key = (r['model'] if r['model'] != 'aquanet_v3'
                   else f"aquanet-{r['head']}-{r['msrb']}-{r['csab']}")
            by[TAG2PRETTY.get(key, key)].setdefault(r['seed'], (r['macro_f1'], stage))

    tuned = {r['name']: r for r in stage_p()}
    tuned_runs = defaultdict(dict)
    for f in sorted(P4.glob('P_*.json')):
        r = json.loads(f.read_text())
        if 'macro_f1' in r:
            tuned_runs[TAG2PRETTY.get(_stem(r['tag']), _stem(r['tag']))][r['seed']] = r['macro_f1']

    out = {}
    for name, seeds in by.items():
        if name not in tuned:
            continue
        matched = sorted(set(seeds) & set(tuned_runs[name]))
        if not matched:
            continue
        sh = np.mean([seeds[s][0] for s in matched])
        tu = np.mean([tuned_runs[name][s] for s in matched])
        out[name] = {'shared_mf1': float(sh), 'tuned_mf1': float(tu),
                     'gain': float(tu - sh), 'n_matched': len(matched), 'seeds': matched}
    return out


def factorial():
    """Balanced Stage B head x MSRB x CSAB factorial: cell means and marginal means.

    Marginals average cell means rather than raw runs. On the balanced design the two agree,
    but averaging cells keeps the estimator correct if a cell ever gains or loses a seed.
    """
    base, aux = _stage_b_runs()
    cells = defaultdict(list)
    for r in base:
        cells[(r['head'], r['msrb'], r['csab'])].append(r['macro_f1'])
    cell_mean = {k: float(np.mean(v)) for k, v in cells.items()}
    cell_n = {k: len(v) for k, v in cells.items()}

    def marginal(axis):
        g = defaultdict(list)
        for k, m in cell_mean.items():
            g[k[axis]].append(m)
        return {lvl: float(np.mean(v)) for lvl, v in g.items()}

    return {'n_runs': len(base), 'n_aux_runs': len(aux), 'n_cells': len(cell_mean),
            'cells': cell_mean, 'cell_n': cell_n,
            'seeds': sorted(set(r['seed'] for r in base)),
            'heads': ['flat', 'hier_naive', 'hier_tf'],
            'msrb': ['off', 'matched', 'on'], 'csab': ['off', 'on'],
            'marginal_head': marginal(0), 'marginal_msrb': marginal(1),
            'marginal_csab': marginal(2),
            'balanced': len(set(cell_n.values())) == 1}


def lambda_sweep():
    """The auxiliary hierarchical-loss mixing sweep, (hier_tf, MSRB on, CSAB on) only."""
    _, aux = _stage_b_runs()
    g = defaultdict(list)
    for r in aux:
        c = r['config']
        if c.get('uncertain_binary') == 'contam':
            g[c.get('lambda_mix')].append(r['macro_f1'])
    base, _ = _stage_b_runs()
    for r in base:
        if (r['head'], r['msrb'], r['csab']) == ('hier_tf', 'on', 'on'):
            g[0.5].append(r['macro_f1'])
    return {'points': [{'lambda_mix': k, 'mf1': float(np.mean(v)),
                        'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 'n': len(v)}
                       for k, v in sorted(g.items())]}


def uncertain_binary_variant():
    """The auxiliary uncertain_binary=exclude runs against their matched base cell."""
    base, aux = _stage_b_runs()
    ex = {r['seed']: r['macro_f1'] for r in aux
          if r['config'].get('uncertain_binary') == 'exclude'}
    inc = {r['seed']: r['macro_f1'] for r in base
           if (r['head'], r['msrb'], r['csab']) == ('hier_tf', 'on', 'on')}
    seeds = sorted(set(ex) & set(inc))
    if not seeds:
        return None
    d = [ex[s] - inc[s] for s in seeds]
    return {'seeds': seeds, 'contam_mf1': float(np.mean([inc[s] for s in seeds])),
            'exclude_mf1': float(np.mean([ex[s] for s in seeds])),
            'mean_diff': float(np.mean(d)), 'wins': int(sum(1 for x in d if x > 0))}


def matched_triplets():
    """Seed-, head- and CSAB-matched MSRB comparisons: on vs matched, and on vs off.

    MatchedNeck is the control that separates "MSRB helps" from "3.4M extra parameters
    help": it carries the same parameter and normalisation count as MSRB with none of the
    multi-scale structure. Only the balanced factorial is used.
    """
    base, _ = _stage_b_runs()
    idx = defaultdict(dict)
    for r in base:
        idx[(r['head'], r['csab'], r['seed'])][r['msrb']] = r['macro_f1']
    out = {}
    for a, b in (('on', 'matched'), ('on', 'off'), ('matched', 'off')):
        d = [v[a] - v[b] for v in idx.values() if a in v and b in v]
        out[f'{a}_vs_{b}'] = {'n': len(d), 'mean_diff': float(np.mean(d)),
                              'wins': int(sum(1 for x in d if x > 0))}
    return out


# ------------------------------------------------------------------ predictions

def predictions(pretty, seed):
    d = json.loads((PRED4 / f'P_{PRETTY2TAG[pretty]}_seed{seed}.json').read_text())
    return (np.array(d['y_true']), np.array(d['y_pred']),
            np.array(d['y_prob']), d['classes'])


def confusion(pretty, seed=None):
    """Confusion matrix summed over all available seeds (or one seed if given)."""
    seeds = [seed] if seed is not None else SEEDS
    m = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for s in seeds:
        yt, yp, _, _ = predictions(pretty, s)
        for t, p in zip(yt, yp):
            m[t, p] += 1
    return m


def per_class_f1():
    """{pretty: {class: {'mean','sd'}}} across the five Stage P seeds."""
    from sklearn.metrics import f1_score
    out = {}
    for name in TAG2PRETTY.values():
        per = defaultdict(list)
        for s in SEEDS:
            yt, yp, _, _ = predictions(name, s)
            f = f1_score(yt, yp, average=None, labels=range(len(CLASSES)), zero_division=0)
            for c, v in zip(CLASSES, f):
                per[c].append(v)
        out[name] = {c: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1))}
                     for c, v in per.items()}
    return out


# ------------------------------------------------------------------ extras

def classical():
    p = path_for('classical')
    return json.loads(p.read_text()) if p.exists() else None


def shortcut():
    """Metadata-only and colour-only probes: how much of D1 needs no real image content."""
    p = path_for('shortcut')
    return json.loads(p.read_text()) if p.exists() else None


def dataset_stats():
    p = path_for('dataset_stats')
    return json.loads(p.read_text()) if p.exists() else None


def dataset_audit():
    p = path_for('dataset_audit')
    return json.loads(p.read_text()) if p.exists() else None


def _reference_run():
    """One Stage P run record, used only for environment and protocol metadata."""
    for q in sorted(P4.glob('P_swin_tiny_seed*.json')) or sorted(P4.glob('P_*.json')):
        return json.loads(q.read_text())
    raise FileNotFoundError(f'no Stage P run records under {P4}')


def env():
    """Torch/CUDA/GPU actually used, taken from a run record, not from this machine."""
    return _reference_run().get('env', {})


def training_config():
    return _reference_run().get('config', {})


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def paired_calibration_tests(metric='ece_raw', reference='AquaNet-no-neck', absolute_from=None):
    """Paired seed-level t tests of a calibration metric, reference vs every other model.

    `absolute_from` compares |x - c| instead of x, which is how the fitted temperature is
    tested: T is better the closer it sits to 1, in either direction.
    """
    from scipy.stats import ttest_rel
    cal = calibration()[metric]
    ref = np.array([cal[reference]['by_seed'][s] for s in SEEDS])
    if absolute_from is not None:
        ref = np.abs(ref - absolute_from)
    rows = []
    for name in cal:
        if name == reference:
            continue
        o = np.array([cal[name]['by_seed'][s] for s in SEEDS])
        if absolute_from is not None:
            o = np.abs(o - absolute_from)
        d = o - ref                       # positive => the reference is better
        rows.append({'name': name, 'mean_diff': float(d.mean()),
                     'wins': int((d > 0).sum()), 'n': len(d),
                     'p': float(ttest_rel(o, ref).pvalue)})
    for r, a in zip(rows, holm([r['p'] for r in rows])):
        r['p_holm'] = float(a)
    return sorted(rows, key=lambda r: r['p'])


def ece_source_agreement():
    """GATE C: the two ECE pipelines are different code paths. Quantify the disagreement."""
    rep = _load('final_report')
    raw = calibration()['ece_raw']
    rows = []
    for s in rep['summary']:
        n = CFG2PRETTY.get(s['config'], s['config'])
        rows.append({'name': n, 'pipeline_ece': s['ece_mean'], 'stageD_ece_raw': raw[n]['mean'],
                     'abs_diff': abs(s['ece_mean'] - raw[n]['mean'])})
    return {'rows': sorted(rows, key=lambda r: -r['abs_diff']),
            'max_abs_diff': max(r['abs_diff'] for r in rows)}
