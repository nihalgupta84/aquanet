"""Stage D: decision-quality evaluation. Inference only -- no training, no new data.

  calibration  ECE / NLL / Brier / reliability bins, temperature scaling fitted on VAL only
  abstention   risk-coverage, AURC, accuracy at fixed coverage; `uncertain` as reject target
  binary       clean/contaminated sensitivity, specificity, FALSE-CLEAN RATE, AUROC
  corruptions  ImageNet-C style sweep generated from the EXISTING test images
  complexity   params, FLOPs, GPU/CPU latency, peak memory

The corruption benchmark doubles as an acquisition-artefact control: JPEG re-encoding
and rescaling destroy resolution and quantisation-table signatures, so graceful
degradation under corruption is evidence a model is not leaning on them
(AQUANET_Q4_PLAN.md section 4.3).
"""
import sys, json, argparse, io, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

from dataset.water_dataset import WaterQualityDataset, CLASSES
from dataset.transforms import get_transforms
from utils.soft_gating import SoftProbabilisticGating
from phase4_helpers import load_checkpoint_models, DEVICE

REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)
PRED4 = ROOT / 'predictions' / 'phase4'
UNCERTAIN = CLASSES.index('uncertain')
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ corruptions

def c_jpeg(im, s):
    q = [50, 35, 25, 15, 8][s - 1]
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=q); buf.seek(0)
    return Image.open(buf).convert('RGB')

def c_blur(im, s):        return im.filter(ImageFilter.GaussianBlur([0.7, 1.4, 2.2, 3.2, 4.5][s - 1]))
def c_motion_blur(im, s):
    k = [3, 5, 7, 9, 13][s - 1]
    a = np.asarray(im).astype(np.float32)
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
    from scipy.ndimage import convolve
    return Image.fromarray(np.stack([convolve(a[..., c], ker, mode='reflect')
                                     for c in range(3)], -1).clip(0, 255).astype(np.uint8))
def c_brightness(im, s):  return ImageEnhance.Brightness(im).enhance([1.25, 1.45, 1.7, 2.0, 2.4][s - 1])
def c_contrast(im, s):    return ImageEnhance.Contrast(im).enhance([0.75, 0.6, 0.45, 0.3, 0.2][s - 1])
def c_gauss_noise(im, s):
    a = np.asarray(im).astype(np.float32)
    a += np.random.default_rng(0).normal(0, [6, 12, 20, 30, 45][s - 1], a.shape)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))
def c_color_temp(im, s):
    a = np.asarray(im).astype(np.float32); f = [0.06, 0.12, 0.18, 0.26, 0.35][s - 1]
    a[..., 0] *= 1 + f; a[..., 2] *= 1 - f
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))
def c_downscale(im, s):
    f = [1.5, 2, 3, 4, 6][s - 1]; w, h = im.size
    return im.resize((max(8, int(w / f)), max(8, int(h / f))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
def c_occlusion(im, s):
    a = np.asarray(im).copy(); h, w = a.shape[:2]
    frac = [0.05, 0.1, 0.17, 0.25, 0.35][s - 1]
    rng = np.random.default_rng(0)
    bh, bw = int(h * frac ** 0.5), int(w * frac ** 0.5)
    y, x = rng.integers(0, max(1, h - bh)), rng.integers(0, max(1, w - bw))
    a[y:y + bh, x:x + bw] = 0
    return Image.fromarray(a)

CORRUPTION_FNS = {'jpeg': c_jpeg, 'blur': c_blur, 'motion_blur': c_motion_blur,
                  'brightness': c_brightness, 'contrast': c_contrast,
                  'gauss_noise': c_gauss_noise, 'color_temp': c_color_temp,
                  'downscale': c_downscale, 'occlusion': c_occlusion}


class CorruptedTestSet(Dataset):
    """Existing test images with a corruption applied. No new data is introduced."""

    def __init__(self, base, corruption=None, severity=0, img_size=224):
        self.samples = base.samples
        self.fn = CORRUPTION_FNS[corruption] if corruption else None
        self.severity = severity
        self.tf = transforms.Compose([transforms.Resize((img_size, img_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        im = Image.open(p).convert('RGB')
        if self.fn:
            im = self.fn(im, self.severity)
        return self.tf(im), y


# ------------------------------------------------------------------ metrics

def ece_bins(prob, y, bins=15):
    conf, pred = prob.max(1)
    acc = pred.eq(y).float()
    e, rows = 0.0, []
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            a, c, w = float(acc[m].mean()), float(conf[m].mean()), float(m.float().mean())
            e += w * abs(a - c)
            rows.append({'bin_lo': float(lo), 'n': int(m.sum()), 'accuracy': a, 'confidence': c})
    return float(e), rows


def temperature_scale(val_logp, val_y):
    """Fit a single temperature on VALIDATION only (RESEARCH_PLAN.md 18: never on test)."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.nll_loss(torch.log_softmax(val_logp / logT.exp(), 1), val_y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp().item())


def risk_coverage(prob, y):
    """Selective prediction. AURC and accuracy at fixed coverage."""
    conf, pred = prob.max(1)
    order = torch.argsort(conf, descending=True)
    correct = pred[order].eq(y[order]).float().numpy()
    n = len(correct)
    cum = np.cumsum(correct)
    cov = np.arange(1, n + 1) / n
    risk = 1 - cum / np.arange(1, n + 1)
    at = {}
    for c in (0.8, 0.9, 0.95, 1.0):
        k = max(1, int(round(c * n)))
        at[f'acc@cov{int(c*100)}'] = float(cum[k - 1] / k)
    return {'aurc': float(np.trapezoid(risk, cov)), **at,
            'curve': [{'coverage': float(cov[i]), 'risk': float(risk[i])}
                      for i in range(0, n, max(1, n // 100))]}


def binary_metrics(prob, y):
    """clean(0) vs contaminated(1..6). False-clean is the costly error for irrigation."""
    p_contam = 1 - prob[:, 0]
    truth = (y > 0).numpy().astype(int)
    pred = (prob.argmax(1) > 0).numpy().astype(int)
    tp = int(((pred == 1) & (truth == 1)).sum()); fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum()); fp = int(((pred == 1) & (truth == 0)).sum())
    out = {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
           'sensitivity_contaminated': tp / max(1, tp + fn),
           'specificity_clean': tn / max(1, tn + fp),
           'false_clean_rate': fn / max(1, tp + fn),
           'n_clean': int((truth == 0).sum()), 'n_contaminated': int((truth == 1).sum())}
    try:
        out['auroc'] = float(roc_auc_score(truth, p_contam.numpy()))
        out['brier'] = float(brier_score_loss(truth, p_contam.numpy()))
    except ValueError:
        out['auroc'] = out['brier'] = None
    out['note'] = (f"only {out['n_clean']} clean images in the test split; specificity and "
                   f"AUROC carry wide CIs and must be reported with them")
    return out


# ------------------------------------------------------------------ tasks

@torch.no_grad()
def infer(model, loader, head, gating):
    model.eval()
    P, Y = [], []
    # WaterQualityDataset yields 4 items (image, label, binary, type); CorruptedTestSet yields 2.
    for batch in loader:
        x, y = batch[0], batch[1]
        out = model(x.to(DEVICE, non_blocking=True))
        if isinstance(out, dict):
            p = torch.softmax(out['flat_logits'], 1) if head == 'flat' else gating(out['binary_logits'], out['type_logits'])
        else:
            p = torch.softmax(out, 1)
        P.append(p.float().cpu()); Y.append(y)
    return torch.cat(P), torch.cat(Y)


def task_calibration(args):
    models = load_checkpoint_models(stage=args.stage)
    base_val = WaterQualityDataset(ROOT / 'data' / 'cleaned_water_dataset', 'val', get_transforms(224, False))
    base_te = WaterQualityDataset(ROOT / 'data' / 'cleaned_water_dataset', 'test', get_transforms(224, False))
    vl = DataLoader(base_val, batch_size=args.batch, num_workers=args.workers)
    tl = DataLoader(base_te, batch_size=args.batch, num_workers=args.workers)
    gating = SoftProbabilisticGating().to(DEVICE)
    out = {}
    for tag, (model, head) in models.items():
        pv, yv = infer(model, vl, head, gating)
        pt, yt = infer(model, tl, head, gating)
        T = temperature_scale(torch.log(pv + 1e-9), yv)
        pt_cal = torch.softmax(torch.log(pt + 1e-9) / T, 1)
        e_raw, bins_raw = ece_bins(pt, yt)
        e_cal, bins_cal = ece_bins(pt_cal, yt)
        onehot = F.one_hot(yt, 7).float()
        out[tag] = {
            'temperature_fitted_on_val': T,
            'ece_raw': e_raw, 'ece_calibrated': e_cal,
            'nll_raw': float(F.nll_loss(torch.log(pt + 1e-9), yt)),
            'nll_calibrated': float(F.nll_loss(torch.log(pt_cal + 1e-9), yt)),
            'brier_raw': float(((pt - onehot) ** 2).sum(1).mean()),
            'brier_calibrated': float(((pt_cal - onehot) ** 2).sum(1).mean()),
            'reliability_raw': bins_raw, 'reliability_calibrated': bins_cal,
        }
        print(f'  {tag:50} T={T:.3f}  ECE {e_raw:.4f} -> {e_cal:.4f}')
        del model; torch.cuda.empty_cache()
    (REPORTS / 'stageD_calibration.json').write_text(json.dumps(out, indent=2))


def _from_saved_predictions(stage=None):
    """Abstention and binary metrics only need stored probabilities; no GPU pass.

    These two tasks read predictions rather than checkpoints, so they honour --stage here
    instead of through load_checkpoint_models. Without the filter they silently report on
    every run ever trained, mixing the shared-LR screening stages into a table that is
    supposed to describe the tuned finalists.
    """
    for p in sorted(PRED4.glob('*.json')):
        r = json.loads(p.read_text())
        if stage and r['meta'].get('stage') != stage:
            continue
        yield r['meta']['tag'], torch.tensor(r['y_prob']), torch.tensor(r['y_true'])


def task_abstention(args):
    out = {}
    for tag, prob, y in _from_saved_predictions(args.stage):
        rc = risk_coverage(prob, y)
        # `uncertain` as the designed reject target (AQUANET_Q4_PLAN.md section 2).
        is_unc = (y == UNCERTAIN)
        conf = prob.max(1).values
        thr = torch.quantile(conf, 0.2)
        rejected = conf <= thr
        out[tag] = {
            **{k: v for k, v in rc.items() if k != 'curve'}, 'curve': rc['curve'],
            'uncertain_reject_precision': float((rejected & is_unc).sum() / max(1, rejected.sum())),
            'uncertain_reject_recall': float((rejected & is_unc).sum() / max(1, is_unc.sum())),
            'reject_threshold_at_20pct_coverage_drop': float(thr),
        }
        print(f"  {tag:50} AURC={rc['aurc']:.4f} acc@90={rc['acc@cov90']:.4f}")
    (REPORTS / 'stageD_abstention.json').write_text(json.dumps(out, indent=2))


def task_binary(args):
    out = {}
    for tag, prob, y in _from_saved_predictions(args.stage):
        out[tag] = binary_metrics(prob, y)
        m = out[tag]
        print(f"  {tag:50} false-clean={m['false_clean_rate']:.4f} "
              f"sens={m['sensitivity_contaminated']:.4f} spec={m['specificity_clean']:.4f}")
    (REPORTS / 'stageD_binary.json').write_text(json.dumps(out, indent=2))


def task_corruptions(args):
    models = load_checkpoint_models(stage=args.stage)
    base = WaterQualityDataset(ROOT / 'data' / 'cleaned_water_dataset', 'test', get_transforms(224, False))
    gating = SoftProbabilisticGating().to(DEVICE)
    corruptions = args.corruptions.split()
    severities = [int(s) for s in args.severities.split()]
    out = {}
    for tag, (model, head) in models.items():
        clean_loader = DataLoader(CorruptedTestSet(base), batch_size=args.batch, num_workers=args.workers)
        p, y = infer(model, clean_loader, head, gating)
        clean_f1 = f1_score(y, p.argmax(1), average='macro')
        rec = {'clean_macro_f1': float(clean_f1), 'corruptions': {}}
        for c in corruptions:
            per_sev = []
            for s in severities:
                ld = DataLoader(CorruptedTestSet(base, c, s), batch_size=args.batch, num_workers=args.workers)
                pc, yc = infer(model, ld, head, gating)
                per_sev.append(float(f1_score(yc, pc.argmax(1), average='macro')))
            rec['corruptions'][c] = {'macro_f1_by_severity': per_sev,
                                     'mean_macro_f1': float(np.mean(per_sev)),
                                     'degradation': float(clean_f1 - np.mean(per_sev))}
            print(f'  {tag:40} {c:14} mean mF1={np.mean(per_sev):.4f} '
                  f'(-{clean_f1 - np.mean(per_sev):.4f})')
        allm = [v['mean_macro_f1'] for v in rec['corruptions'].values()]
        rec['mean_corruption_macro_f1'] = float(np.mean(allm))
        rec['mean_degradation'] = float(clean_f1 - np.mean(allm))
        out[tag] = rec
        del model; torch.cuda.empty_cache()
    (REPORTS / 'stageD_corruptions.json').write_text(json.dumps(out, indent=2))


def task_complexity(args):
    models = load_checkpoint_models(stage=args.stage)
    gating = SoftProbabilisticGating().to(DEVICE)
    out = {}
    for tag, (model, head) in models.items():
        params = sum(p.numel() for p in model.parameters())
        rec = {'params_m': params / 1e6}
        try:
            from fvcore.nn import FlopCountAnalysis
            rec['gflops'] = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).to(DEVICE)).total() / 1e9
        except Exception as e:
            rec['gflops'] = None; rec['gflops_note'] = f'fvcore unavailable ({type(e).__name__})'
        for dev in ('cuda', 'cpu'):
            m = model.to(dev).eval()
            x = torch.randn(1, 3, 224, 224, device=dev)
            with torch.no_grad():
                for _ in range(10):
                    m(x)
                if dev == 'cuda':
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(50):
                    m(x)
                if dev == 'cuda':
                    torch.cuda.synchronize()
                rec[f'latency_ms_{dev}'] = (time.perf_counter() - t0) * 1000 / 50
            if dev == 'cuda':
                rec['peak_mem_mib'] = torch.cuda.max_memory_allocated() / 2**20
        out[tag] = rec
        print(f"  {tag:50} {rec['params_m']:.2f}M  gpu {rec['latency_ms_cuda']:.2f}ms  "
              f"cpu {rec['latency_ms_cpu']:.2f}ms")
        del model; torch.cuda.empty_cache()
    (REPORTS / 'stageD_complexity.json').write_text(json.dumps(out, indent=2))


TASKS = {'calibration': task_calibration, 'abstention': task_abstention, 'binary': task_binary,
         'corruptions': task_corruptions, 'complexity': task_complexity}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=sorted(TASKS))
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--corruptions', default=' '.join(CORRUPTION_FNS))
    ap.add_argument('--severities', default='1 2 3 4 5')
    # Restrict to one training stage. Stage P is the tuned 5-seed finalist set and is the
    # only thing the paper reports; sweeping A/B/C as well costs hours and evaluates
    # screening checkpoints trained under the shared-LR protocol that Stage T replaced.
    ap.add_argument('--stage', default=None, help="e.g. P; omit to use every checkpoint")
    a = ap.parse_args()
    print(f'[stageD] task={a.task}')
    TASKS[a.task](a)
    print(f'[stageD] {a.task} done')


if __name__ == '__main__':
    main()
