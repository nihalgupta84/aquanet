"""Shared checkpoint loading for the Stage D/E/F evaluation scripts."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from models.proposed.aquanet_v3 import AquaNetV3
from models.deep_learning.dl_baselines import get_dl_baseline_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
P4 = ROOT / 'phase4_results'
CKPT4 = ROOT / 'checkpoints' / 'phase4'


def build_from_config(cfg):
    """Rebuild the exact architecture a phase 4 run used, from its saved config."""
    from experiments.phase4_pipeline import MatchedNeck
    model_name = cfg['model']
    if model_name != 'aquanet_v3':
        from experiments.phase4_pipeline import TIMM_ALIAS
        return get_dl_baseline_model(TIMM_ALIAS.get(model_name, model_name), 7, False), 'flat'
    m = AquaNetV3(7, pretrained=False, use_msrb=(cfg['msrb'] == 'on'), use_csab=(cfg['csab'] == 'on'))
    if cfg['msrb'] == 'matched':
        m.msrb = MatchedNeck(1024, 512)
    return m, cfg['head']


def load_checkpoint_models(stage=None, limit=None, require_seeds=None):
    """Yield {tag: (model_on_device, head)} for every phase 4 run with a checkpoint.

    Loads lazily-ish: the caller is expected to free each model after use.
    """
    out = {}
    for res in sorted(P4.glob('*.json')):
        try:
            r = json.loads(res.read_text())
        except json.JSONDecodeError:
            continue
        if 'config' not in r or 'tag' not in r:
            continue
        if stage and r.get('stage') != stage:
            continue
        if require_seeds and r['seed'] not in require_seeds:
            continue
        ck = CKPT4 / f"{r['tag']}.pth"
        if not ck.exists():
            print(f"  [!] {r['tag']}: checkpoint missing, skipped")
            continue
        model, head = build_from_config(r['config'])
        model.load_state_dict(torch.load(ck, map_location='cpu'))
        out[r['tag']] = (model.to(DEVICE).eval(), head)
        if limit and len(out) >= limit:
            break
    if not out:
        print('  [!] no phase 4 checkpoints found -- run the training stages first')
    return out
