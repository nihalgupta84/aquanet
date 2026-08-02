"""Backfill per-image predictions for phase 3 runs that finished before dumping was added.

RESEARCH_PLAN.md section 9 requires per-image IDs, labels, probabilities and predictions
for McNemar tests and prediction-level bootstrap. `train_one` never wrote them, so the
12 completed runs have only scalar metrics. This re-runs inference from the saved
checkpoints -- it does not retrain anything, and it verifies that the reproduced test
metrics match the recorded ones before writing.
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.water_dataset import WaterQualityDataset, CLASSES
from dataset.transforms import get_transforms
from models.proposed.aquanet_v3 import AquaNetV3
from models.deep_learning.dl_baselines import get_dl_baseline_model
from utils.soft_gating import SoftProbabilisticGating

OUT = ROOT / 'phase3_results'
CKPT = ROOT / 'checkpoints' / 'phase3'
PRED = ROOT / 'predictions' / 'phase3'
PRED.mkdir(parents=True, exist_ok=True)
TOL = 1e-3


def make_model(model, variant):
    if model == 'aquanet_v3':
        return AquaNetV3(7, True, use_msrb=variant != 'no_msrb', use_csab=variant != 'no_csab')
    return get_dl_baseline_model(model, 7, True)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = WaterQualityDataset(ROOT / 'data' / 'cleaned_water_dataset', 'test', get_transforms(224, False))
    # workers=0: order must exactly match ds.samples, and this runs once.
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
    paths = [str(Path(s[0]).relative_to(ROOT)) for s in ds.samples]
    gating = SoftProbabilisticGating().to(device)

    done = skipped = failed = 0
    for res in sorted(OUT.glob('*_seed*.json')):
        rec = json.loads(res.read_text())
        if 'model' not in rec:
            continue
        name, variant, seed = rec['model'], rec['variant'], rec['seed']
        tag = f'{name}_{variant}_seed{seed}'
        if (PRED / f'{tag}.json').exists():
            skipped += 1
            continue
        ck = CKPT / f'{tag}.pth'
        if not ck.exists():
            print(f'  [!] {tag}: no checkpoint at {ck.relative_to(ROOT)} -- cannot backfill; '
                  f'this run must be repeated to satisfy RESEARCH_PLAN.md section 9')
            failed += 1
            continue

        model = make_model(name, variant).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()

        P, Y = [], []
        with torch.no_grad():
            for x, y, *_ in loader:
                out = model(x.to(device))
                if name == 'aquanet_v3':
                    p = (torch.softmax(out['flat_logits'], 1) if variant == 'flat'
                         else gating(out['binary_logits'], out['type_logits']))
                else:
                    p = torch.softmax(out, 1)
                P.append(p.cpu()); Y.append(y)
        p = torch.cat(P); y = torch.cat(Y)

        acc = float(y.eq(p.argmax(1)).float().mean())
        delta = abs(acc - rec['accuracy'])
        status = 'ok' if delta < TOL else f'MISMATCH (recorded {rec["accuracy"]:.4f})'
        if delta >= TOL:
            print(f'  [!] {tag}: reproduced accuracy {acc:.4f} != recorded {rec["accuracy"]:.4f}. '
                  f'Writing anyway, but this run is not reproducible from its checkpoint.')

        (PRED / f'{tag}.json').write_text(json.dumps({
            'meta': {'model': name, 'variant': variant, 'seed': seed, 'protocol': 'phase3',
                     'backfilled': True, 'reproduced_accuracy': acc,
                     'recorded_accuracy': rec['accuracy']},
            'classes': list(CLASSES), 'image_path': paths,
            'y_true': [int(v) for v in y.tolist()],
            'y_pred': [int(v) for v in p.argmax(1).tolist()],
            'y_prob': [[round(float(v), 6) for v in r] for r in p.tolist()],
        }))
        print(f'  {tag}: acc={acc:.4f} {status}')
        done += 1
        del model
        torch.cuda.empty_cache()

    print(f'backfill: {done} written, {skipped} already present, {failed} missing checkpoints')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
