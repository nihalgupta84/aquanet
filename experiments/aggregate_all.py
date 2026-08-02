"""Aggregation, decision gates and the final statistical report.

Implements RESEARCH_PLAN.md section 9:
  - >= 5 independent seeds, mean, sample SD, 95% CI
  - prediction-level bootstrap differences
  - McNemar from paired per-image predictions
  - paired seed-level tests with effect sizes
  - Holm correction across multiple baseline comparisons

and RESEARCH_PLAN.md section 8: candidates are ranked on VALIDATION macro-F1.
Test numbers are reported, never used to choose.
"""
import sys, json, argparse, itertools
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import ttest_rel, wilcoxon, binomtest

P3, P4 = ROOT / 'phase3_results', ROOT / 'phase4_results'
PRED3, PRED4 = ROOT / 'predictions' / 'phase3', ROOT / 'predictions' / 'phase4'
RNG = np.random.default_rng(20260731)


# ------------------------------------------------------------------ loading

def load_runs(d, stage=None):
    runs = []
    if not d.exists():
        return runs
    for p in sorted(d.glob('*.json')):
        if p.name in ('phase3_summary.json', 'dataset_audit.json'):
            continue
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if 'accuracy' not in r:
            continue
        if stage and r.get('stage') != stage:
            continue
        r['_file'] = p.name
        runs.append(r)
    return runs


def config_key(r):
    """Group runs that differ only by seed."""
    if 'head' in r:
        k = f"aquanet[{r['head']},msrb={r['msrb']},csab={r['csab']}]" if r['model'] == 'aquanet_v3' else r['model']
        cfg = r.get('config', {})
        if cfg.get('uncertain_binary') == 'exclude':
            k += '+uexcl'
        if r['head'] != 'flat' and cfg.get('lambda_mix', 0.5) != 0.5:
            k += f"+lam{cfg['lambda_mix']}"
        return k
    return f"{r['model']}/{r['variant']}"


def val_f1_at_selected(r):
    """Validation macro-F1 of the epoch that was actually checkpointed."""
    h = r.get('history') or []
    if not h:
        return None
    if 'val_macro_f1' in h[0] and 'selected_epoch' in r:          # phase 4
        for e in h:
            if e['epoch'] == r['selected_epoch']:
                return e['val_macro_f1']
    if 'val_loss' in h[0]:                                        # phase 3: selected on NLL
        return min(h, key=lambda e: e['val_loss'])['val_macro_f1']
    return max(h, key=lambda e: e['val_macro_f1'])['val_macro_f1']


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float('nan'), float('nan'))
    se = x.std(ddof=1) / np.sqrt(len(x))
    from scipy.stats import t
    h = se * t.ppf(0.975, len(x) - 1)
    return (float(x.mean() - h), float(x.mean() + h))


def summarise(runs):
    g = defaultdict(list)
    for r in runs:
        g[config_key(r)].append(r)
    out = []
    for k, rs in g.items():
        acc = [r['accuracy'] for r in rs]
        f1 = [r['macro_f1'] for r in rs]
        vf1 = [v for v in (val_f1_at_selected(r) for r in rs) if v is not None]
        out.append({
            'config': k, 'n_seeds': len(rs), 'seeds': sorted(r['seed'] for r in rs),
            'val_macro_f1_mean': float(np.mean(vf1)) if vf1 else None,
            'test_acc_mean': float(np.mean(acc)),
            'test_acc_sd': float(np.std(acc, ddof=1)) if len(acc) > 1 else None,
            'test_acc_ci95': ci95(acc),
            'test_macro_f1_mean': float(np.mean(f1)),
            'test_macro_f1_sd': float(np.std(f1, ddof=1)) if len(f1) > 1 else None,
            'test_macro_f1_ci95': ci95(f1),
            'ece_mean': float(np.mean([r['ece_15bin'] for r in rs])),
            'params_m': rs[0].get('params_m'),
            'runs': rs,
        })
    return sorted(out, key=lambda z: -z['test_macro_f1_mean'])


def table(summ, title, sort_by='test_macro_f1_mean'):
    print(f'\n{title}')
    print(f"{'config':44} {'n':>2} {'VAL mF1':>8} {'TEST acc':>9} {'sd':>7} {'TEST mF1':>9} {'sd':>7} {'params':>8}")
    print('-' * 105)
    for s in sorted(summ, key=lambda z: -(z[sort_by] or 0)):
        v = f"{s['val_macro_f1_mean']:.4f}" if s['val_macro_f1_mean'] is not None else '   --  '
        sda = f"{s['test_acc_sd']:.4f}" if s['test_acc_sd'] is not None else '  --  '
        sdf = f"{s['test_macro_f1_sd']:.4f}" if s['test_macro_f1_sd'] is not None else '  --  '
        pm = f"{s['params_m']:.2f}M" if s['params_m'] else '   --'
        print(f"{s['config']:44} {s['n_seeds']:>2} {v:>8} {s['test_acc_mean']:>9.4f} {sda:>7} "
              f"{s['test_macro_f1_mean']:>9.4f} {sdf:>7} {pm:>8}")


# ------------------------------------------------------------------ statistics

def paired_seed_test(a_runs, b_runs, metric='accuracy'):
    """Paired seed-level test with Cohen's dz. Requires matched seeds."""
    a = {r['seed']: r[metric] for r in a_runs}
    b = {r['seed']: r[metric] for r in b_runs}
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    x = np.array([a[s] for s in seeds]); y = np.array([b[s] for s in seeds])
    d = x - y
    res = {'n_seeds': len(seeds), 'seeds': seeds, 'mean_diff': float(d.mean()),
           'wins': int((d > 0).sum()),
           'cohens_dz': float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float('inf')}
    res['paired_t_p'] = float(ttest_rel(x, y).pvalue)
    if len(seeds) >= 6:
        res['wilcoxon_p'] = float(wilcoxon(x, y).pvalue)
    else:
        # Exact Wilcoxon on n<6 cannot reach p<0.05 regardless of effect size.
        res['wilcoxon_p'] = None
        res['wilcoxon_note'] = f'not reportable at n={len(seeds)}; min attainable two-sided p is {2**-(len(seeds)-1):.3f}'
    return res


def load_preds(tag):
    for d in (PRED4, PRED3):
        p = d / f'{tag}.json'
        if p.exists():
            return json.loads(p.read_text())
    return None


def mcnemar(tag_a, tag_b):
    """Exact McNemar from paired per-image predictions (RESEARCH_PLAN.md section 9)."""
    A, B = load_preds(tag_a), load_preds(tag_b)
    if A is None or B is None:
        return None
    if A['image_path'] != B['image_path']:
        return {'error': 'image order differs between runs; cannot pair'}
    ya = np.array(A['y_pred']) == np.array(A['y_true'])
    yb = np.array(B['y_pred']) == np.array(B['y_true'])
    n01 = int((~ya & yb).sum()); n10 = int((ya & ~yb).sum())
    if n01 + n10 == 0:
        return {'n01': 0, 'n10': 0, 'p': 1.0, 'note': 'identical predictions'}
    p = binomtest(n10, n01 + n10, 0.5).pvalue
    return {'n01_b_only': n01, 'n10_a_only': n10, 'p': float(p)}


def bootstrap_diff(tag_a, tag_b, n=10000, metric='macro_f1'):
    """Prediction-level bootstrap of the metric difference (RESEARCH_PLAN.md section 9)."""
    from sklearn.metrics import f1_score, accuracy_score
    A, B = load_preds(tag_a), load_preds(tag_b)
    if A is None or B is None or A['image_path'] != B['image_path']:
        return None
    y = np.array(A['y_true']); pa = np.array(A['y_pred']); pb = np.array(B['y_pred'])
    fn = (lambda t, p: f1_score(t, p, average='macro')) if metric == 'macro_f1' else accuracy_score
    idx = RNG.integers(0, len(y), size=(n, len(y)))
    d = np.array([fn(y[i], pa[i]) - fn(y[i], pb[i]) for i in idx])
    return {'metric': metric, 'mean_diff': float(d.mean()),
            'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            'p_two_sided': float(2 * min((d <= 0).mean(), (d >= 0).mean())), 'n_boot': n}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ gates and reports

def phase3_gate():
    runs = load_runs(P3)
    summ = summarise(runs)
    table(summ, 'Phase 3 -- all completed runs (selection rule ACTUALLY used: validation NLL)')

    by = {s['config']: s for s in summ}
    flat = by.get('aquanet_v3/flat')
    print('\nGATE (AQUANET_Q4_PLAN.md section 3, Stage 0):')
    if not flat:
        print('  aquanet_v3/flat has no results -- gate cannot be read'); return 2
    if flat['n_seeds'] < 3:
        print(f"  aquanet_v3/flat still has only {flat['n_seeds']} seed(s) {flat['seeds']}. "
              f"Stage 0 is incomplete; run ./scripts/run_all.sh stage0")
        return 1

    base = max((by[k]['test_acc_mean'] for k in by if k.split('/')[0] in ('resnet50', 'mobilenetv2')),
               default=0.0)
    print(f"  aquanet_v3/flat  3-seed test accuracy = {flat['test_acc_mean']:.4f} "
          f"(sd {flat['test_acc_sd']:.4f}, seeds {flat['seeds']})")
    print(f"  best baseline    3-seed test accuracy = {base:.4f}")
    if flat['test_acc_mean'] >= 0.87:
        print(f"  {'PASS':>6}: flat holds above 0.87. The seed-42 result was not a lucky draw.")
        print(f"          -> proceed to Stage A (./scripts/run_all.sh stageA)")
    elif flat['test_acc_mean'] > base:
        print(f"  {'PARTIAL':>6}: flat beats the baselines but fell below 0.87. Proceed to Stage A,")
        print(f"          and expect the head analysis to carry more of the paper than the raw margin.")
    else:
        print(f"  {'FAIL':>6}: flat does not beat the baselines across seeds. The seed-42 run was a")
        print(f"          lucky draw. Reorder: run Stage B before Stage A -- the paper's centre")
        print(f"          becomes the head analysis (AQUANET_Q4_PLAN.md section 2), not the margin.")

    print('\nSelection-rule disagreement (AQUANET_Q4_PLAN.md 1-D1):')
    print(f"{'config':30} {'val mF1 @ckpt':>14} {'val mF1 best':>13} {'lost':>8}")
    for s in summ:
        best = [max(r['history'], key=lambda e: e['val_macro_f1'])['val_macro_f1']
                for r in s['runs'] if r.get('history')]
        if not best or s['val_macro_f1_mean'] is None:
            continue
        b = float(np.mean(best))
        print(f"{s['config']:30} {s['val_macro_f1_mean']:>14.4f} {b:>13.4f} "
              f"{s['val_macro_f1_mean'] - b:>8.4f}")
    return 0


def stage_report(stage):
    runs = load_runs(P4, stage=stage)
    if not runs:
        print(f'no phase 4 runs found for stage {stage}'); return 1
    summ = summarise(runs)
    table(summ, f'Stage {stage} -- ranked by test macro-F1 (selection is on VAL macro-F1, shown left)')
    print(f'\nRESEARCH_PLAN.md 8.1 selection (validation macro-F1 only):')
    best = max(summ, key=lambda s: s['val_macro_f1_mean'] or -1)
    print(f"  candidate = {best['config']}  (val macro-F1 {best['val_macro_f1_mean']:.4f})")
    return 0


def final_report(outdir, stage=None):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # Scope to one stage. Pooling every stage averages a config's Stage T sweep runs --
    # deliberately mistuned points from the LR grid -- into its headline mean, which both
    # depresses the mean and inflates n_seeds with repeats of the same seed.
    runs = load_runs(P4, stage)
    if not runs:
        print(f'no phase 4 runs yet{f" for stage {stage}" if stage else ""}'); return 1
    summ = summarise(runs)
    table(summ, f'PHASE 4 RUNS{f" -- STAGE {stage}" if stage else ""}')

    # Selection: validation only, five seeds only.
    eligible = [s for s in summ if s['n_seeds'] >= 5 and s['val_macro_f1_mean'] is not None]
    if not eligible:
        print('\nNo configuration has the >=5 seeds RESEARCH_PLAN.md section 9 requires. '
              'Report is provisional.')
        eligible = [s for s in summ if s['val_macro_f1_mean'] is not None]
    chosen = max(eligible, key=lambda s: s['val_macro_f1_mean'])
    print(f"\nFINAL MODEL (RESEARCH_PLAN.md 8.1, validation macro-F1 only): {chosen['config']}")
    print(f"  val macro-F1 {chosen['val_macro_f1_mean']:.4f} | "
          f"test acc {chosen['test_acc_mean']:.4f} CI95 {tuple(round(v,4) for v in chosen['test_acc_ci95'])} | "
          f"test macro-F1 {chosen['test_macro_f1_mean']:.4f} CI95 {tuple(round(v,4) for v in chosen['test_macro_f1_ci95'])}")

    by = {s['config']: s for s in summ}
    comparisons, pvals = [], []
    for s in summ:
        if s['config'] == chosen['config']:
            continue
        for metric in ('accuracy', 'macro_f1'):
            t = paired_seed_test(chosen['runs'], s['runs'], metric)
            if t:
                comparisons.append({'vs': s['config'], 'metric': metric, **t})
                pvals.append(t['paired_t_p'])
    if pvals:
        for c, a in zip(comparisons, holm(pvals)):
            c['paired_t_p_holm'] = float(a)

    print('\nPaired seed-level comparisons vs the selected model (Holm-corrected):')
    print(f"{'vs':40} {'metric':>9} {'n':>2} {'delta':>9} {'wins':>6} {'dz':>7} {'p':>8} {'p_holm':>8}")
    print('-' * 96)
    for c in sorted(comparisons, key=lambda z: z['paired_t_p']):
        print(f"{c['vs']:40} {c['metric']:>9} {c['n_seeds']:>2} {c['mean_diff']:>+9.4f} "
              f"{c['wins']}/{c['n_seeds']:<4} {c['cohens_dz']:>7.2f} {c['paired_t_p']:>8.4f} "
              f"{c.get('paired_t_p_holm', float('nan')):>8.4f}")

    # Prediction-level: McNemar + bootstrap on the best seed of each config.
    print('\nPrediction-level tests (seed-matched, best available seed):')
    pred_tests = []
    for s in summ:
        if s['config'] == chosen['config']:
            continue
        seeds = sorted(set(r['seed'] for r in chosen['runs']) & set(r['seed'] for r in s['runs']))
        if not seeds:
            continue
        seed = seeds[0]
        ta = next(r['tag'] for r in chosen['runs'] if r['seed'] == seed)
        tb = next(r['tag'] for r in s['runs'] if r['seed'] == seed)
        mc, bs = mcnemar(ta, tb), bootstrap_diff(ta, tb)
        if mc and bs:
            pred_tests.append({'vs': s['config'], 'seed': seed, 'mcnemar': mc, 'bootstrap': bs})
            print(f"  vs {s['config']:38} seed {seed:<5} McNemar p={mc.get('p', float('nan')):.4f}  "
                  f"bootstrap dF1={bs['mean_diff']:+.4f} CI95[{bs['ci95'][0]:+.4f},{bs['ci95'][1]:+.4f}]")

    payload = {'summary': [{k: v for k, v in s.items() if k != 'runs'} for s in summ],
               'selected': chosen['config'], 'paired_seed_tests': comparisons,
               'prediction_level_tests': pred_tests}
    (outdir / 'final_report.json').write_text(json.dumps(payload, indent=2, default=str))
    print(f'\nwritten: {outdir / "final_report.json"}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase3-gate', action='store_true')
    ap.add_argument('--stage')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--out', default='reports/final')
    a = ap.parse_args()
    if a.phase3_gate:
        return phase3_gate()
    if a.report:
        return final_report(a.out, a.stage)   # --stage scopes the report, if given
    if a.stage:
        return stage_report(a.stage)
    return phase3_gate()


if __name__ == '__main__':
    sys.exit(main())
