"""Stage E: generalisation using data already on disk.

  D2 = data/cleaned_scrapper_finetune     (213 images)  adaptation
  D3 = data/cleaned_scrapper_unseen_test  (146 images)  held-out evaluation

RESEARCH_PLAN.md section 12 / contribution C4. Never used in the phase 3 study.

D3 has 14-26 images per class, so per-class numbers are unstable by construction.
Everything here is reported with bootstrap CIs and must be described in the paper as a
generalisation indication, not a benchmark (AQUANET_Q4_PLAN.md section 3, Stage E).
"""
import sys, json, argparse, copy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score

from dataset.water_dataset import WaterQualityDataset, CLASSES
from dataset.transforms import get_transforms
from utils.seed import set_seed
from utils.soft_gating import SoftProbabilisticGating
from phase4_helpers import load_checkpoint_models, build_from_config, DEVICE, P4, CKPT4

REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)
D2 = ROOT / 'data' / 'cleaned_scrapper_finetune'
D3 = ROOT / 'data' / 'cleaned_scrapper_unseen_test'
RNG = np.random.default_rng(20260731)


def bootstrap_ci(y, p, fn, n=5000):
    idx = RNG.integers(0, len(y), size=(n, len(y)))
    vals = np.array([fn(y[i], p[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


@torch.no_grad()
def infer(model, loader, head, gating):
    model.eval()
    P, Y = [], []
    for batch in loader:
        out = model(batch[0].to(DEVICE))
        if isinstance(out, dict):
            p = torch.softmax(out['flat_logits'], 1) if head == 'flat' else gating(out['binary_logits'], out['type_logits'])
        else:
            p = torch.softmax(out, 1)
        P.append(p.float().cpu()); Y.append(batch[1])
    return torch.cat(P), torch.cat(Y)


def scored(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    mf1 = f1_score(y, pred, average='macro')
    acc = accuracy_score(y, pred)
    per = {}
    for i, c in enumerate(CLASSES):
        m = y == i
        per[c] = {'n': int(m.sum()),
                  'recall': float((pred[m] == i).mean()) if m.any() else None}
    return {
        'accuracy': float(acc),
        'accuracy_ci95': bootstrap_ci(y, pred, accuracy_score),
        'macro_f1': float(mf1),
        'macro_f1_ci95': bootstrap_ci(y, pred, lambda a, b: f1_score(a, b, average='macro')),
        'per_class': per,
        'binary_false_clean_rate': float(((pred == 0) & (y > 0)).sum() / max(1, (y > 0).sum())),
    }


def task_zeroshot(args):
    """D1-trained models evaluated on D3 without any adaptation."""
    models = load_checkpoint_models(stage=args.stage)
    ds = WaterQualityDataset(D3, 'test', get_transforms(224, False))
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers)
    gating = SoftProbabilisticGating().to(DEVICE)
    out = {}
    for tag, (model, head) in models.items():
        p, y = infer(model, loader, head, gating)
        out[tag] = scored(y.numpy(), p.argmax(1).numpy())
        print(f"  {tag:50} D3 macro-F1={out[tag]['macro_f1']:.4f} "
              f"CI95{tuple(round(v,3) for v in out[tag]['macro_f1_ci95'])} "
              f"acc={out[tag]['accuracy']:.4f}")
        del model; torch.cuda.empty_cache()
    out['_note'] = ('D3 = 146 images, 14-26 per class. Report as a generalisation indication '
                    'with CIs, not as a benchmark.')
    (REPORTS / 'stageE_zeroshot.json').write_text(json.dumps(out, indent=2))


def pick_best_run(stage=None):
    """The candidate selected on VALIDATION macro-F1 (RESEARCH_PLAN.md 8.1)."""
    best, best_v = None, -1.0
    for res in sorted(P4.glob('*.json')):
        try:
            r = json.loads(res.read_text())
        except json.JSONDecodeError:
            continue
        if 'best_val_macro_f1' not in r:
            continue
        if stage and r.get('stage') != stage:
            continue
        if r['best_val_macro_f1'] > best_v:
            best_v, best = r['best_val_macro_f1'], r
    return best


def task_adapt(args):
    """Fine-tune the selected D1 model on a fraction of D2, evaluate on D3."""
    run = pick_best_run(args.stage)
    if run is None:
        print('  [!] no phase 4 runs available to adapt; run the training stages first')
        return
    tag = f"E_adapt_{args.fraction}pct_seed{args.seed}"
    outfile = REPORTS / f'stageE_adapt_{args.fraction}pct_seed{args.seed}.json'
    if outfile.exists():
        print(f'  [skip] {outfile.name}'); return

    set_seed(args.seed)
    model, head = build_from_config(run['config'])
    model.load_state_dict(torch.load(CKPT4 / f"{run['tag']}.pth", map_location='cpu'))
    model = model.to(DEVICE)
    gating = SoftProbabilisticGating().to(DEVICE)

    d2 = WaterQualityDataset(D2, 'train', get_transforms(224, True))
    n = int(len(d2) * args.fraction / 100)
    idx = RNG.permutation(len(d2))[:n]
    sub = Subset(d2, idx.tolist())
    tr = DataLoader(sub, batch_size=min(args.batch, max(2, n)), shuffle=True,
                    num_workers=args.workers, drop_last=False)
    d3 = DataLoader(WaterQualityDataset(D3, 'test', get_transforms(224, False)),
                    batch_size=args.eval_batch, num_workers=args.workers)

    p0, y0 = infer(model, d3, head, gating)
    before = scored(y0.numpy(), p0.argmax(1).numpy())

    # Low LR, few epochs: D2 is 213 images. This is adaptation, not retraining.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
    for epoch in range(args.adapt_epochs):
        model.train()
        for batch in tr:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            out = model(x)
            logits = out['flat_logits'] if isinstance(out, dict) else out
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    p1, y1 = infer(model, d3, head, gating)
    after = scored(y1.numpy(), p1.argmax(1).numpy())
    print(f"  {tag}: D3 macro-F1 {before['macro_f1']:.4f} -> {after['macro_f1']:.4f} "
          f"(n_adapt={n})")
    outfile.write_text(json.dumps({
        'tag': tag, 'source_run': run['tag'], 'fraction_pct': args.fraction,
        'n_adaptation_images': n, 'seed': args.seed,
        'adapt_epochs': args.adapt_epochs, 'before': before, 'after': after,
    }, indent=2))


def task_summarise(args):
    rows = []
    for p in sorted(REPORTS.glob('stageE_adapt_*.json')):
        r = json.loads(p.read_text())
        rows.append((r['fraction_pct'], r['seed'], r['before']['macro_f1'], r['after']['macro_f1']))
    if not rows:
        print('  no adaptation results yet'); return
    print(f"\n{'D2 fraction':>12} {'n seeds':>8} {'D3 mF1 before':>14} {'D3 mF1 after':>13} {'delta':>8}")
    for frac in sorted(set(r[0] for r in rows)):
        sel = [r for r in rows if r[0] == frac]
        b = np.mean([r[2] for r in sel]); a = np.mean([r[3] for r in sel])
        print(f'{frac:>11}% {len(sel):>8} {b:>14.4f} {a:>13.4f} {a-b:>+8.4f}')
    print('\nD3 = 146 images. These are generalisation indications with wide CIs, not benchmarks.')


TASKS = {'zeroshot': task_zeroshot, 'adapt': task_adapt, 'summarise': task_summarise}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=sorted(TASKS))
    ap.add_argument('--fraction', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--adapt-epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--eval-batch', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    # accepted and ignored so the stage can share p4_common flags
    ap.add_argument('--epochs', type=int, default=None); ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--accum', type=int, default=None); ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--backbone-lr-mult', type=float, default=None); ap.add_argument('--warmup-epochs', type=int, default=None)
    ap.add_argument('--clip-grad', type=float, default=None); ap.add_argument('--loss', default=None)
    ap.add_argument('--focal-gamma', type=float, default=None); ap.add_argument('--balance', default=None)
    ap.add_argument('--amp', type=int, default=1); ap.add_argument('--channels-last', type=int, default=1)
    ap.add_argument('--stage', default=None, help="e.g. P; omit to use every checkpoint")
    a = ap.parse_args()
    print(f'[stageE] task={a.task}')
    TASKS[a.task](a)


if __name__ == '__main__':
    main()
