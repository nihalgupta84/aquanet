"""Per-model learning-rate selection on VALIDATION macro-F1 only.

RESEARCH_PLAN.md section 7 requires a *comparable tuning budget* per model, not identical
hyperparameters. Stages A-C used one shared (lr, backbone_lr_mult) for every model, which
handed AquaNet an advantage: AquaNet has ~3.5M freshly-initialised parameters training at
the full rate, while ResNet50 is almost entirely pretrained and therefore ran at 3e-5.
ResNet50 ended up the worst model in the study at val macro-F1 0.8234, below the 0.8672 it
reached under the phase 3 protocol -- an artefact of our own configuration, not of ResNet50.

Stage T gives every model the same budget: |LR_GRID| x |MULT_GRID| configurations at one
seed, each selected on validation macro-F1. This script reads those runs back and reports
the winner per model. Test data is never consulted.

Usage:
    python experiments/select_lr.py --table          # what won, for the paper's appendix
    python experiments/select_lr.py --get resnet50   # "3e-04 1.0", consumed by run_all.sh
"""
import sys, json, argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / 'phase4_results'


def sweep_results():
    """{model_key: [(val_macro_f1, lr, backbone_lr_mult, tag), ...]} from Stage T runs."""
    out = defaultdict(list)
    for f in sorted(P4.glob('T_*.json')):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        cfg = r.get('config', {})
        lr, mult = cfg.get('lr'), cfg.get('backbone_lr_mult')
        if lr is None or mult is None or 'best_val_macro_f1' not in r:
            continue
        out[model_key(r)].append((r['best_val_macro_f1'], lr, mult, r['tag']))
    return out


def model_key(r):
    if r['model'] != 'aquanet_v3':
        return r['model']
    return f"aquanet-{r['head']}-{r['msrb']}-{r['csab']}"


def best(rows):
    return max(rows)  # tuples compare on val macro-F1 first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', action='store_true')
    ap.add_argument('--get', help='print "<lr> <mult>" for this model key and exit')
    a = ap.parse_args()
    res = sweep_results()

    if a.get:
        if a.get not in res:
            # No sweep yet: fall back to the shared defaults so the caller still runs.
            print('3e-4 0.1')
            return 0
        _, lr, mult, _ = best(res[a.get])
        print(f'{lr:g} {mult:g}')
        return 0

    if not res:
        print('  no Stage T runs found -- run ./scripts/run_all.sh stageT first')
        return 1

    print(f"\n{'model':<28} {'n cfgs':>6} {'best LR':>9} {'bb mult':>8} {'VAL mF1':>9} "
          f"{'worst':>9} {'spread':>8}")
    print('-' * 84)
    for k in sorted(res, key=lambda k: -best(res[k])[0]):
        rows = res[k]
        v, lr, mult, _ = best(rows)
        lo = min(r[0] for r in rows)
        print(f'{k:<28} {len(rows):>6} {lr:>9.1e} {mult:>8.2f} {v:>9.4f} {lo:>9.4f} '
              f'{v - lo:>+8.4f}')
    print('\nSelection is on validation macro-F1 only; test data is not consulted.')
    print('"spread" is how much the shared-hyperparameter protocol could have cost this model.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
