"""Phase 4 trainer: one run per invocation, protocol-repaired.

Fixes the defects recorded in AQUANET_Q4_PLAN.md section 1:
  D1  checkpoint on validation macro-F1 (RESEARCH_PLAN.md 8.1), not validation NLL
  D2  one loss function applied identically to every model
  D3  parameter groups (backbone vs new modules) + warmup + cosine
  D4  class balance corrected once, not twice
  D5  head and neck are independent axes, so the factorial in Stage B is constructible
  D6  per-image predictions written for every run (RESEARCH_PLAN.md 9)

Driven by scripts/run_all.sh. One process = one (model, head, msrb, csab, seed) run.
"""
import sys, argparse, json, os, platform, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score

from dataset.water_dataset import WaterQualityDataset, CLASSES
from dataset.transforms import get_transforms
from models.proposed.aquanet_v3 import AquaNetV3
from models.deep_learning.dl_baselines import get_dl_baseline_model
from utils.metrics import compute_metrics
from utils.seed import set_seed
from utils.soft_gating import SoftProbabilisticGating

OUT = ROOT / 'phase4_results'
CKPT = ROOT / 'checkpoints' / 'phase4'
PRED = ROOT / 'predictions' / 'phase4'
for d in (OUT, CKPT, PRED):
    d.mkdir(parents=True, exist_ok=True)

TIMM_ALIAS = {
    'mobilenetv2': 'mobilenetv2_100',
    'convnext_tiny': 'convnext_tiny',
    'swin_tiny': 'swin_tiny_patch4_window7_224',
    'deit_small': 'deit_small_patch16_224',
    'vit_small': 'vit_small_patch16_224',
}


# --------------------------------------------------------------------------- neck controls

class MatchedNeck(nn.Module):
    """Parameter-matched control for MSRB (AQUANET_Q4_PLAN.md section 2, Stage B).

    MSRB(1024->512) holds 3,407,872 conv weights and 3,072 BN affine parameters:
        1x1 branches b1+b4   2 * 1024*128        =   262,144
        3x3 branches b2+b3   2 * 1024*128*9      = 2,359,296
        fuse 1x1             512*512             =   262,144
        shortcut 1x1         1024*512            =   524,288

    This control holds exactly the same:
        1x1 stem             1024*512            =   524,288
        3x3 body             512*512*9           = 2,359,296
        shortcut 1x1         1024*512            =   524,288

    Same parameter count, same BN count, same depth, same receptive-field cost --
    but single-scale and undilated. So "MSRB helps" can be separated from
    "3.4M extra parameters help", which the `no_msrb` ablation cannot do
    (it deletes 2.9M parameters along with the block).
    """

    def __init__(self, in_channels=1024, out_channels=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(self.stem(x)) + self.shortcut(x))


def build_model(args):
    if args.model != 'aquanet_v3':
        return get_dl_baseline_model(TIMM_ALIAS.get(args.model, args.model), 7, not args.scratch)
    m = AquaNetV3(7, pretrained=not args.scratch,
                  use_msrb=(args.msrb == 'on'), use_csab=(args.csab == 'on'))
    if args.msrb == 'matched':
        m.msrb = MatchedNeck(1024, 512)
    return m


# --------------------------------------------------------------------------- heads and loss

def forward_probs(model, x, args, gating):
    """Return 7-class probabilities under the configured head."""
    out = model(x)
    if args.model != 'aquanet_v3':
        return torch.softmax(out, 1)
    if args.head == 'flat':
        return torch.softmax(out['flat_logits'], 1)
    return gating(out['binary_logits'], out['type_logits'])


def classification_loss(logits, y, args, cls_w):
    """D2: identical objective for every model. Chosen once on validation, applied to all."""
    if args.loss == 'ce':
        return F.cross_entropy(logits, y)
    if args.loss == 'wce':
        return F.cross_entropy(logits, y, weight=cls_w)
    if args.loss == 'focal':
        ce = F.cross_entropy(logits, y, reduction='none', weight=cls_w)
        pt = torch.exp(-ce)
        return (((1 - pt) ** args.focal_gamma) * ce).mean()
    raise ValueError(args.loss)


def compute_loss(model, x, y, args, gating, cls_w):
    out = model(x)
    if args.model != 'aquanet_v3':
        return classification_loss(out, y, args, cls_w)

    flat = classification_loss(out['flat_logits'], y, args, cls_w)
    if args.head == 'flat':
        return flat

    b_logits, t_logits = out['binary_logits'], out['type_logits']
    contaminated = (y > 0)

    if args.head == 'hier_naive':
        # The published formulation: NLL over the gating product P(contam)*P(type).
        # Reproduced exactly so the negative result in AQUANET_Q4_PLAN.md 2 stays runnable.
        p7 = gating(b_logits, t_logits)
        hier = F.nll_loss(torch.log(p7 + 1e-7), y, weight=cls_w)
        return args.lambda_mix * hier + (1 - args.lambda_mix) * flat

    if args.head == 'hier_tf':
        # Teacher-forced conditional training (AQUANET_Q4_PLAN.md 2a).
        # The type head is supervised directly on the ground-truth contaminated
        # subset, so its gradient is no longer scaled by P(contaminated) -- which
        # is ~0.5 at init and starves it while the easy binary head converges.
        b_target = contaminated.float().unsqueeze(1)
        if args.uncertain_binary == 'exclude':
            # AQUANET_Q4_PLAN.md 2b: `uncertain` is filed under contaminated/, asserting
            # "the annotator could not tell => the water is contaminated" for 11.4% of
            # train and 16.4% of test. Drop those images from the binary objective only;
            # they still train the type head and are still scored at test time.
            keep = (y != CLASSES.index('uncertain')).float().unsqueeze(1)
        else:
            keep = torch.ones_like(b_target)
        if keep.sum() > 0:
            bce = (F.binary_cross_entropy_with_logits(b_logits, b_target, reduction='none') * keep).sum() / keep.sum()
        else:
            bce = b_logits.sum() * 0.0

        if contaminated.any():
            t_loss = F.cross_entropy(t_logits[contaminated], (y[contaminated] - 1),
                                     weight=cls_w[1:] if cls_w is not None else None)
        else:
            t_loss = t_logits.sum() * 0.0

        hier = bce + t_loss
        return args.lambda_mix * hier + (1 - args.lambda_mix) * flat

    raise ValueError(args.head)


# --------------------------------------------------------------------------- data

def build_loaders(args):
    root = ROOT / 'data' / 'cleaned_water_dataset'
    tr_ds = WaterQualityDataset(root, 'train', get_transforms(args.img_size, True))
    va_ds = WaterQualityDataset(root, 'val', get_transforms(args.img_size, False))
    te_ds = WaterQualityDataset(root, 'test', get_transforms(args.img_size, False))

    # D4: balance ONCE. phase3 used sampler AND loss weights, an approximately squared
    # correction with `oil` at 107 images against `algae` at 695.
    sampler = None
    if args.balance in ('sampler', 'both'):
        counts = np.bincount(np.array(tr_ds.labels), minlength=7)
        w = 1.0 / (counts + 1e-5)
        sampler = WeightedRandomSampler(torch.as_tensor(w[np.array(tr_ds.labels)], dtype=torch.double),
                                        len(tr_ds.labels), replacement=True)

    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0,
                  prefetch_factor=4 if args.workers > 0 else None)
    tr = DataLoader(tr_ds, batch_size=args.batch, sampler=sampler,
                    shuffle=(sampler is None), drop_last=False, **common)
    va = DataLoader(va_ds, batch_size=args.eval_batch, shuffle=False, **common)
    te = DataLoader(te_ds, batch_size=args.eval_batch, shuffle=False, **common)
    return tr_ds, va_ds, te_ds, tr, va, te


def class_weights(tr_ds, args, device):
    if args.balance not in ('weights', 'both'):
        return None
    counts = np.bincount(np.array(tr_ds.labels), minlength=7)
    w = torch.tensor(1.0 / (counts + 1e-5), dtype=torch.float32, device=device)
    # Normalise to mean 1, not sum 1. phase3 used sum-1, silently scaling the loss
    # by ~1/7 and interacting with ReduceLROnPlateau's absolute threshold.
    return w / w.mean()


# --------------------------------------------------------------------------- optimiser

def build_optimizer(model, args):
    """D3: pretrained trunk and freshly-initialised modules get different learning rates.

    AquaNet stacks ~3.5M randomly-initialised parameters (MSRB+CSAB+FC+heads) onto a
    pretrained DenseNet121. ResNet50 has ~0.014M. A single lr therefore penalises
    AquaNet uniquely -- and is the most likely reason deleting MSRB "helped" in phase 3.
    """
    backbone, fresh = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if n.startswith('backbone.') else fresh).append(p)
    if not backbone:  # timm baselines: everything is pretrained except the classifier
        head_names = ('fc.', 'classifier.', 'head.')
        backbone, fresh = [], []
        for n, p in model.named_parameters():
            (fresh if any(h in n for h in head_names) else backbone).append(p)
    groups = [{'params': backbone, 'lr': args.lr * args.backbone_lr_mult},
              {'params': fresh, 'lr': args.lr}]
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def lr_lambda(args, steps_per_epoch):
    warm = max(1, args.warmup_epochs * steps_per_epoch)
    total = max(warm + 1, args.epochs * steps_per_epoch)

    def fn(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
    return fn


# --------------------------------------------------------------------------- eval

def ece(prob, y, bins=15):
    conf, pred = prob.max(1)
    val = 0.0
    for lo in torch.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            val += m.float().mean() * abs(pred[m].eq(y[m]).float().mean() - conf[m].mean())
    return float(val)


@torch.no_grad()
def evaluate(model, loader, args, gating, device, timed=False):
    model.eval()
    P, Y, lat = [], [], []
    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1]
        if timed:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.amp.autocast('cuda', enabled=args.amp):
            p = forward_probs(model, x, args, gating)
        if timed:
            torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1000 / len(x))
        P.append(p.float().cpu()); Y.append(y)
    return torch.cat(P), torch.cat(Y), (float(np.mean(lat)) if lat else None)


def env_metadata():
    def sh(c):
        try:
            return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'python': platform.python_version(), 'git_commit': sh('git rev-parse HEAD'),
            'git_dirty': bool(sh('git status --porcelain'))}


# --------------------------------------------------------------------------- run

def run(args):
    tag = args.tag
    result_path = OUT / f'{tag}.json'
    if result_path.exists() and not args.force:
        print(f'[skip] {result_path.name} already exists', flush=True)
        return

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda')

    tr_ds, va_ds, te_ds, tr, va, te = build_loaders(args)
    model = build_model(args).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    gating = SoftProbabilisticGating().to(device)
    cls_w = class_weights(tr_ds, args, device)
    opt = build_optimizer(model, args)
    steps = max(1, (len(tr) + args.accum - 1) // args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(args, steps))
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    ckpt = CKPT / f'{tag}.pth'
    best, best_epoch, bad, history = -1.0, 0, 0, []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(tr):
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda', enabled=args.amp):
                loss = compute_loss(model, x, y, args, gating, cls_w) / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0 or (i + 1) == len(tr):
                if args.clip_grad:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sched.step()

        p, y, _ = evaluate(model, va, args, gating, device)
        vf1 = f1_score(y, p.argmax(1), average='macro')
        vacc = float(y.eq(p.argmax(1)).float().mean())
        vnll = float(F.nll_loss(torch.log(p + 1e-7), y))
        history.append({'epoch': epoch, 'val_macro_f1': float(vf1), 'val_accuracy': vacc,
                        'val_nll': vnll, 'lr': opt.param_groups[-1]['lr']})
        # D1: RESEARCH_PLAN.md 8.1 -- select on validation macro-F1. Never on test, never on NLL.
        marker = ''
        if vf1 > best:
            best, best_epoch, bad = float(vf1), epoch, 0
            torch.save(model.state_dict(), ckpt); marker = ' *'
        else:
            bad += 1
        print(f'[{tag}] epoch={epoch}/{args.epochs} val_macro_f1={vf1:.4f} val_acc={vacc:.4f} '
              f'val_nll={vnll:.4f}{marker}', flush=True)
        if args.patience and bad >= args.patience:
            print(f'[{tag}] early stop at epoch {epoch} (no val macro-F1 gain in {bad})', flush=True)
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    p, y, lat = evaluate(model, te, args, gating, device, timed=True)
    m = compute_metrics(y, p.argmax(1), CLASSES)
    m.update({
        'tag': tag, 'model': args.model, 'head': args.head, 'msrb': args.msrb, 'csab': args.csab,
        'seed': args.seed, 'stage': args.stage,
        'ece_15bin': ece(p, y), 'nll': float(F.nll_loss(torch.log(p + 1e-7), y)),
        'latency_ms_per_image': lat,
        'params_m': sum(q.numel() for q in model.parameters()) / 1e6,
        'trainable_params_m': sum(q.numel() for q in model.parameters() if q.requires_grad) / 1e6,
        'selected_epoch': best_epoch, 'best_val_macro_f1': best,
        'epochs_run': len(history), 'train_minutes': (time.time() - t_start) / 60,
        'history': history, 'config': vars(args), 'env': env_metadata(),
    })
    result_path.write_text(json.dumps(m, indent=2, default=str))

    # RESEARCH_PLAN.md 9: per-image records for McNemar and prediction-level bootstrap.
    paths = [str(Path(s[0]).relative_to(ROOT)) for s in te_ds.samples]
    assert len(paths) == len(y), 'eval loader order does not match dataset order'
    (PRED / f'{tag}.json').write_text(json.dumps({
        'meta': {'tag': tag, 'model': args.model, 'head': args.head, 'msrb': args.msrb,
                 'csab': args.csab, 'seed': args.seed, 'stage': args.stage, 'protocol': 'phase4'},
        'classes': list(CLASSES), 'image_path': paths,
        'y_true': [int(v) for v in y.tolist()],
        'y_pred': [int(v) for v in p.argmax(1).tolist()],
        'y_prob': [[round(float(v), 6) for v in r] for r in p.tolist()],
    }))
    print(f'[{tag}] DONE  test_acc={m["accuracy"]:.4f}  test_macro_f1={m["macro_f1"]:.4f}  '
          f'params={m["params_m"]:.2f}M  {m["train_minutes"]:.1f}min', flush=True)


def build_parser():
    ap = argparse.ArgumentParser(description='Phase 4 single-run trainer')
    ap.add_argument('--model', default='aquanet_v3')
    ap.add_argument('--head', default='flat', choices=['flat', 'hier_naive', 'hier_tf'])
    ap.add_argument('--msrb', default='on', choices=['on', 'off', 'matched'])
    ap.add_argument('--csab', default='on', choices=['on', 'off'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--stage', default='A')
    ap.add_argument('--tag', default=None)

    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=0, help='0 disables early stopping')
    ap.add_argument('--batch', type=int, default=64, help='physical batch; effective = batch*accum')
    ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--eval-batch', type=int, default=256)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--img-size', type=int, default=224)

    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--backbone-lr-mult', type=float, default=0.1)
    ap.add_argument('--warmup-epochs', type=int, default=3)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--clip-grad', type=float, default=1.0)

    ap.add_argument('--loss', default='wce', choices=['ce', 'wce', 'focal'])
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    ap.add_argument('--balance', default='weights', choices=['sampler', 'weights', 'both', 'none'])
    ap.add_argument('--lambda-mix', type=float, default=0.5)
    ap.add_argument('--uncertain-binary', default='contam', choices=['contam', 'exclude'])

    ap.add_argument('--amp', type=int, default=1)
    ap.add_argument('--channels-last', type=int, default=1)
    ap.add_argument('--scratch', action='store_true')
    ap.add_argument('--force', action='store_true')
    return ap


def main():
    args = build_parser().parse_args()
    if args.tag is None:
        if args.model == 'aquanet_v3':
            args.tag = f'{args.stage}_aquanet_{args.head}_msrb-{args.msrb}_csab-{args.csab}_seed{args.seed}'
        else:
            args.tag = f'{args.stage}_{args.model}_seed{args.seed}'
        if args.uncertain_binary == 'exclude':
            args.tag += '_uexcl'
        if args.lambda_mix != 0.5 and args.head != 'flat':
            args.tag += f'_lam{args.lambda_mix}'
    args.amp = bool(args.amp); args.channels_last = bool(args.channels_last)
    try:
        run(args)
    except torch.cuda.OutOfMemoryError:
        # Deliberately NOT retrying at a smaller batch. Batch size changes the gradient,
        # so a silent fallback would make this run incomparable to its seed-matched peers.
        print(f'[{args.tag}] CUDA OOM at batch={args.batch}. Lower BATCH in scripts/config.sh '
              f'(and raise ACCUM to keep the effective batch fixed), then rerun the whole stage.',
              file=sys.stderr, flush=True)
        sys.exit(75)


if __name__ == '__main__':
    main()
