"""Stage F: interpretability with a quantitative check.

RESEARCH_PLAN.md section 14 and 18 both forbid publishing saliency images with
unsupported claims -- in particular, do not label averaged feature activations as
Grad-CAM and do not claim glare suppression from pictures alone. So this produces:

  gradcam    Grad-CAM and Grad-CAM++ heatmaps (real gradients, not activations)
  deletion   deletion/insertion curves, which is the number that makes the section
             defensible: AUC of accuracy as the most-salient pixels are removed

Both are self-contained -- no external CAM dependency.
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.water_dataset import WaterQualityDataset, CLASSES
from dataset.transforms import get_transforms
from phase4_helpers import load_checkpoint_models, DEVICE

REPORTS = ROOT / 'reports'
FIGS = REPORTS / 'stageF_figures'
REPORTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def target_layer(model):
    """Last spatial conv feature map before pooling."""
    if hasattr(model, 'csab'):                      # AquaNet: after the neck
        return model.csab if not isinstance(model.csab, torch.nn.Identity) else model.msrb
    cands = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    return cands[-1]


class CAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, 'acts', o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'grads', go[0]))

    def _logits(self, x):
        out = self.model(x)
        return out['flat_logits'] if isinstance(out, dict) else out

    def __call__(self, x, cls=None, plusplus=False):
        self.model.zero_grad(set_to_none=True)
        logits = self._logits(x)
        if cls is None:
            cls = logits.argmax(1)
        logits.gather(1, cls.view(-1, 1)).sum().backward()
        a, g = self.acts, self.grads
        if plusplus:
            g2, g3 = g ** 2, g ** 3
            denom = 2 * g2 + (a * g3).sum((2, 3), keepdim=True)
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            w = (alpha * F.relu(g)).sum((2, 3), keepdim=True)
        else:
            w = g.mean((2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        flat = cam.flatten(1)
        lo = flat.min(1, keepdim=True).values.view(-1, 1, 1, 1)
        hi = flat.max(1, keepdim=True).values.view(-1, 1, 1, 1)
        return ((cam - lo) / (hi - lo + 1e-8)).squeeze(1).detach()


def task_gradcam(args):
    models = load_checkpoint_models(stage=args.stage)
    ds = WaterQualityDataset(ROOT / 'data' / 'cleaned_water_dataset', 'test', get_transforms(224, False))
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=args.workers)
    out = {}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        can_plot = True
    except ImportError:
        can_plot = False
        print('  matplotlib unavailable; writing heatmap arrays only')

    for tag, (model, head) in models.items():
        cam = CAM(model, target_layer(model))
        batch = next(iter(loader))   # dataset yields (image, label, binary, type)
        x, y = batch[0].to(DEVICE), batch[1]
        h1 = cam(x, plusplus=False).cpu()
        h2 = cam(x, plusplus=True).cpu()
        out[tag] = {'method': 'Grad-CAM and Grad-CAM++ (true gradients w.r.t. the target logit)',
                    'layer': type(target_layer(model)).__name__,
                    'n_examples': int(len(x)),
                    'mean_activation_area': float((h1 > 0.5).float().mean())}
        if can_plot:
            img = (x.cpu() * STD + MEAN).clamp(0, 1)
            fig, ax = plt.subplots(3, 8, figsize=(20, 8))
            for i in range(min(8, len(x))):
                ax[0, i].imshow(img[i].permute(1, 2, 0)); ax[0, i].set_title(CLASSES[y[i]], fontsize=8)
                ax[1, i].imshow(img[i].permute(1, 2, 0)); ax[1, i].imshow(h1[i], cmap='jet', alpha=0.5)
                ax[2, i].imshow(img[i].permute(1, 2, 0)); ax[2, i].imshow(h2[i], cmap='jet', alpha=0.5)
            for a_ in ax.ravel():
                a_.axis('off')
            ax[1, 0].set_ylabel('Grad-CAM'); ax[2, 0].set_ylabel('Grad-CAM++')
            fig.suptitle(tag); fig.tight_layout()
            fig.savefig(FIGS / f'{tag}_gradcam.png', dpi=130); plt.close(fig)
        print(f'  {tag:50} heatmaps written')
        del model; torch.cuda.empty_cache()
    (REPORTS / 'stageF_gradcam.json').write_text(json.dumps(out, indent=2))


def task_deletion(args):
    """Deletion/insertion: the quantitative claim. Lower deletion AUC = better localisation."""
    models = load_checkpoint_models(stage=args.stage)
    ds = WaterQualityDataset(ROOT / 'data' / 'cleaned_water_dataset', 'test', get_transforms(224, False))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    steps = args.steps
    out = {}
    for tag, (model, head) in models.items():
        cam = CAM(model, target_layer(model))
        del_curve = np.zeros(steps + 1)
        ins_curve = np.zeros(steps + 1)
        n_batches = 0
        for bi, (x, y, *_) in enumerate(loader):
            if bi >= args.max_batches:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            h = cam(x)
            order = h.flatten(1).argsort(dim=1, descending=True)
            npix = order.shape[1]
            blur = F.avg_pool2d(x, 15, stride=1, padding=7)
            for s in range(steps + 1):
                k = int(npix * s / steps)
                mask = torch.ones_like(h).flatten(1)
                if k:
                    mask.scatter_(1, order[:, :k], 0.0)
                mask = mask.view_as(h).unsqueeze(1)
                with torch.no_grad():
                    o_del = model(x * mask + blur * (1 - mask))
                    o_ins = model(x * (1 - mask) + blur * mask)
                    for o, curve in ((o_del, del_curve), (o_ins, ins_curve)):
                        lg = o['flat_logits'] if isinstance(o, dict) else o
                        curve[s] += float(lg.argmax(1).eq(y).float().mean())
            n_batches += 1
        del_curve /= max(1, n_batches); ins_curve /= max(1, n_batches)
        out[tag] = {
            'deletion_curve': del_curve.tolist(), 'insertion_curve': ins_curve.tolist(),
            'deletion_auc': float(np.trapezoid(del_curve, dx=1 / steps)),
            'insertion_auc': float(np.trapezoid(ins_curve, dx=1 / steps)),
            'n_images': n_batches * args.batch,
            'interpretation': 'lower deletion AUC and higher insertion AUC indicate the '
                              'saliency map localises evidence the model actually uses',
        }
        print(f"  {tag:50} deletion AUC={out[tag]['deletion_auc']:.4f} "
              f"insertion AUC={out[tag]['insertion_auc']:.4f}")
        del model; torch.cuda.empty_cache()
    (REPORTS / 'stageF_deletion.json').write_text(json.dumps(out, indent=2))


TASKS = {'gradcam': task_gradcam, 'deletion': task_deletion}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=sorted(TASKS))
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--max-batches', type=int, default=8)
    ap.add_argument('--stage', default=None, help="e.g. P; omit to use every checkpoint")
    a = ap.parse_args()
    print(f'[stageF] task={a.task}')
    TASKS[a.task](a)


if __name__ == '__main__':
    main()
