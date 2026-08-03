"""How much of D1 is solvable without looking at the picture?

AUDIT.md finding F1 records that the class label is confounded with the image source. This
script makes that finding a reproducible artefact instead of a shell snippet, because it is
the evidence behind the paper's Dataset Contribution Analysis.

Two probes, both trained on the D1 train split and scored on the D1 test split:

  metadata  a random forest over file metadata only -- width, height, aspect ratio, pixel
            count, file size, bytes per pixel, and the JPEG quantisation tables. No pixel
            content of any kind reaches the classifier.
  colour    mean and standard deviation of the three RGB channels, six numbers per image.
            A floor for "how far does average colour alone get you".

Both are compared with the majority-class baseline. A metadata probe far above the majority
baseline means source identity leaks the label, so absolute accuracy on this dataset is
inflated relative to any field deployment. Every model sees identical inputs and splits, so
the relative ranking the paper claims is unaffected -- but the reader must be told.

Usage:
    python experiments/shortcut_probe.py --out reports/stageH_shortcut.json
"""
import sys, json, argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'cleaned_water_dataset'
CLASSES = ['clean', 'algae', 'debris', 'foam', 'oil', 'turbid', 'uncertain']
SEEDS = [7, 21, 42, 1337, 2024]
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def paths(split):
    out = []
    for ci, c in enumerate(CLASSES):
        d = DATA / split / 'clean' if c == 'clean' else DATA / split / 'contaminated' / c
        for p in sorted(d.rglob('*')):
            if p.is_file() and p.suffix.lower() in EXT:
                out.append((p, ci))
    return out


def features(split):
    meta, colour, y = [], [], []
    for p, ci in paths(split):
        with Image.open(p) as im:
            w, h = im.size
            q = getattr(im, 'quantization', None) or {}
            rgb = np.asarray(im.convert('RGB').resize((32, 32)), dtype=np.float32)
        fs = p.stat().st_size
        meta.append([w, h, w / h, w * h, fs, fs / (w * h),
                     sum(sum(v) for v in q.values()), len(q)])
        colour.append(np.concatenate([rgb.mean((0, 1)), rgb.std((0, 1))]))
        y.append(ci)
    return np.array(meta, float), np.array(colour, float), np.array(y)


def run_probe(Xtr, ytr, Xte, yte, name):
    accs, f1s, cms = [], [], []
    for s in SEEDS:
        p = RandomForestClassifier(400, random_state=s, n_jobs=16).fit(Xtr, ytr).predict(Xte)
        accs.append(accuracy_score(yte, p))
        f1s.append(f1_score(yte, p, average='macro', zero_division=0))
        cms.append(confusion_matrix(yte, p, labels=range(len(CLASSES))))
    per = f1_score(yte, RandomForestClassifier(400, random_state=42, n_jobs=16)
                   .fit(Xtr, ytr).predict(Xte), average=None,
                   labels=range(len(CLASSES)), zero_division=0)
    return {
        'probe': name, 'n_features': int(Xtr.shape[1]),
        'accuracy_mean': float(np.mean(accs)), 'accuracy_sd': float(np.std(accs, ddof=1)),
        'macro_f1_mean': float(np.mean(f1s)), 'macro_f1_sd': float(np.std(f1s, ddof=1)),
        'per_class_f1': {c: float(v) for c, v in zip(CLASSES, per)},
        'confusion_matrix_seed42': cms[SEEDS.index(42)].tolist(), 'seeds': SEEDS,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='reports/stageH_shortcut.json')
    a = ap.parse_args()

    print('extracting metadata and colour features')
    Mtr, Ctr, ytr = features('train')
    Mte, Cte, yte = features('test')
    print(f'  train {len(ytr)}  test {len(yte)}')

    maj = Counter(ytr).most_common(1)[0][0]
    majority = {
        'strategy': f'always predict "{CLASSES[maj]}" (most frequent training class)',
        'accuracy': float((yte == maj).mean()),
        'macro_f1': float(f1_score(yte, np.full_like(yte, maj), average='macro',
                                   zero_division=0)),
    }

    out = {'_meta': {
        'classes': CLASSES, 'n_train': int(len(ytr)), 'n_test': int(len(yte)),
        'note': ('The metadata probe never sees pixel content. Its margin over the '
                 'majority baseline measures how far source identity alone predicts the '
                 'label on this dataset. All models share these splits, so the relative '
                 'ranking reported in the paper is unaffected; absolute accuracy is not '
                 'transferable to a field deployment.'),
    }, 'majority_baseline': majority, 'probes': {}}

    for X, Xt, name in ((Mtr, Mte, 'metadata_only'), (Ctr, Cte, 'mean_colour_only')):
        r = run_probe(X, ytr, Xt, yte, name)
        out['probes'][name] = r
        print(f"  {name:18} acc {r['accuracy_mean']:.4f} +- {r['accuracy_sd']:.4f}  "
              f"macro-F1 {r['macro_f1_mean']:.4f}")
    print(f"  {'majority':18} acc {majority['accuracy']:.4f}  "
          f"macro-F1 {majority['macro_f1']:.4f}")

    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f'written: {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
