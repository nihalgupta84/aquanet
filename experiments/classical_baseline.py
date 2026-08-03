"""Classical (non-deep) baselines on the same D1 splits, same selection rule.

RESEARCH_PLAN.md section 17 lists a classical ML benchmark. Its purpose here is not to
compete with the CNNs -- it is to establish how much of the task is solvable from generic
colour/texture statistics alone, which bounds how much credit any architecture can claim.

Protocol mirrors the deep pipeline exactly where it can:
  - identical train/val/test image lists (1956/416/427), never re-split
  - identical 224x224 aspect-ratio-distorting resize (see transforms note in the paper's
    Limitations: the distortion is a known confound, reproduced here deliberately so the
    classical and deep feature pipelines see the same pixels)
  - class weights balanced and normalised to mean 1, as in the deep loss
  - EQUAL TUNING BUDGET: six trials per model, selected on VALIDATION macro-F1 only.
    This is the same budget every deep model received in Stage T. Test is never consulted.
  - the promoted configuration is then refit under seeds {7,21,42,1337,2024}

Features: RGB + HSV histograms, LBP (two scales), GLCM Haralick descriptors, HOG.

Usage:
    python experiments/classical_baseline.py --out reports/stageG_classical.json
"""
import sys, json, argparse, time
from pathlib import Path

import numpy as np
import cv2
from joblib import Parallel, delayed
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'cleaned_water_dataset'
CLASSES = ['clean', 'algae', 'debris', 'foam', 'oil', 'turbid', 'uncertain']
SEEDS = [7, 21, 42, 1337, 2024]
GLCM_PROPS = ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM')
XGB_THREADS = 16


# ------------------------------------------------------------------ data

def split_index(split):
    """(paths, labels) for one split, in the directory layout the deep loader uses."""
    root = DATA / split
    items = []
    for ci, c in enumerate(CLASSES):
        d = root / 'clean' if c == 'clean' else root / 'contaminated' / c
        for p in sorted(d.rglob('*')):
            if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                items.append((str(p), ci))
    paths = [i[0] for i in items]
    y = np.array([i[1] for i in items], dtype=np.int64)
    return paths, y


# ------------------------------------------------------------------ features

def features_one(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    # Same resize as transforms.py: aspect ratio is NOT preserved. Deliberate -- see docstring.
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = []
    # Colour histograms, L1-normalised so image size cannot leak in.
    for src, chans, rngs in ((img, (0, 1, 2), ((0, 256),) * 3),
                             (hsv, (0, 1, 2), ((0, 180), (0, 256), (0, 256)))):
        for ch, rng in zip(chans, rngs):
            h = cv2.calcHist([src], [ch], None, [32], list(rng)).flatten()
            f.append(h / (h.sum() + 1e-8))
    # Channel moments.
    f.append(np.concatenate([img.mean((0, 1)), img.std((0, 1)), hsv.mean((0, 1)), hsv.std((0, 1))]))

    # LBP at two scales, uniform patterns.
    for P, R in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        h, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2))
        f.append(h / (h.sum() + 1e-8))

    # GLCM Haralick descriptors, 32 grey levels, 2 distances x 4 angles.
    q = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    f.append(np.concatenate([graycoprops(glcm, p).flatten() for p in GLCM_PROPS]))

    # HOG on a downscaled grey image; coarse cells keep the dimension comparable to the rest.
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    f.append(hog(small, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2),
                 block_norm='L2-Hys', feature_vector=True))

    return np.concatenate(f).astype(np.float32)


FEATURE_DIM = 6 * 32 + 12 + 10 + 18 + len(GLCM_PROPS) * 8 + 3 * 3 * 4 * 9


def extract(paths, jobs):
    t0 = time.time()
    X = np.vstack(Parallel(n_jobs=jobs, batch_size=32)(delayed(features_one)(p) for p in paths))
    print(f'  {len(paths)} images -> {X.shape} in {time.time() - t0:.1f}s', flush=True)
    return X


# ------------------------------------------------------------------ models

def class_weights(y):
    """Balanced weights normalised to mean 1, matching the deep pipeline's loss weighting."""
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = len(y) / (len(CLASSES) * np.maximum(counts, 1))
    return w / w.mean()


def grids():
    """Six trials per model -- the same tuning budget every deep model got in Stage T."""
    return {
        'logreg': [{'C': c} for c in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)],
        'random_forest': [{'n_estimators': n, 'max_features': m}
                          for n in (300, 600) for m in ('sqrt', 'log2', 0.2)],
        'svm_rbf': [{'C': c, 'gamma': g} for c in (1.0, 10.0, 100.0) for g in ('scale', 'auto')],
        'xgboost': [{'max_depth': d, 'learning_rate': lr}
                    for d in (4, 6, 8) for lr in (0.05, 0.1)],
    }


def build(name, hp, y_train, seed):
    w = class_weights(y_train)
    cw = {i: float(v) for i, v in enumerate(w)}
    if name == 'logreg':
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=hp['C'], max_iter=3000, class_weight=cw,
                                                random_state=seed, n_jobs=-1))
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=hp['n_estimators'],
                                      max_features=hp['max_features'], class_weight=cw,
                                      random_state=seed, n_jobs=-1)
    if name == 'svm_rbf':
        return make_pipeline(StandardScaler(),
                             SVC(C=hp['C'], gamma=hp['gamma'], kernel='rbf', class_weight=cw,
                                 probability=False, random_state=seed))
    if name == 'xgboost':
        # n_jobs is capped deliberately. On a 256-core host `n_jobs=-1` spawns hundreds of
        # threads over a 1956-row problem and spends all of its time in synchronisation:
        # measured 1053 s per trial unbounded against ~20 s at 16 threads, same result.
        return XGBClassifier(max_depth=hp['max_depth'], learning_rate=hp['learning_rate'],
                             n_estimators=400, subsample=0.8, colsample_bytree=0.8,
                             tree_method='hist', num_class=len(CLASSES),
                             objective='multi:softprob', random_state=seed,
                             n_jobs=XGB_THREADS, eval_metric='mlogloss')
    raise ValueError(name)


def fit_predict(name, hp, Xtr, ytr, Xev, seed):
    m = build(name, hp, ytr, seed)
    if name == 'xgboost':
        w = class_weights(ytr)
        m.fit(Xtr, ytr, sample_weight=w[ytr])
    else:
        m.fit(Xtr, ytr)
    return m.predict(Xev)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='reports/stageG_classical.json')
    ap.add_argument('--jobs', type=int, default=16)
    a = ap.parse_args()

    print('indexing splits')
    tr_p, ytr = split_index('train')
    va_p, yva = split_index('val')
    te_p, yte = split_index('test')
    print(f'  train {len(ytr)}  val {len(yva)}  test {len(yte)}')
    assert (len(ytr), len(yva), len(yte)) == (1956, 416, 427), 'split sizes must match the paper'

    print('extracting features')
    Xtr, Xva, Xte = (extract(p, a.jobs) for p in (tr_p, va_p, te_p))

    out = {'_meta': {
        'n_train': len(ytr), 'n_val': len(yva), 'n_test': len(yte),
        'classes': CLASSES, 'feature_dim': int(Xtr.shape[1]), 'seeds': SEEDS,
        'trials_per_model': 6,
        'note': ('Selection on validation macro-F1 only, six trials per model -- the same '
                 'tuning budget the deep models received in Stage T. Test never consulted. '
                 'LogisticRegression and SVC are deterministic given fixed data, so their '
                 'across-seed sd is 0 by construction rather than by stability.'),
    }, 'models': {}}

    for name, grid in grids().items():
        print(f'\n=== {name} ===', flush=True)
        trials = []
        for hp in grid:
            t0 = time.time()
            pv = fit_predict(name, hp, Xtr, ytr, Xva, seed=42)
            f1 = f1_score(yva, pv, average='macro')
            trials.append({'hp': hp, 'val_macro_f1': float(f1),
                           'val_accuracy': float(accuracy_score(yva, pv))})
            print(f'  {hp} -> val mF1 {f1:.4f}  ({time.time() - t0:.1f}s)', flush=True)

        best = max(trials, key=lambda t: t['val_macro_f1'])
        print(f'  selected {best["hp"]}  val mF1 {best["val_macro_f1"]:.4f}', flush=True)

        seed_runs = []
        for s in SEEDS:
            pt = fit_predict(name, best['hp'], Xtr, ytr, Xte, seed=s)
            seed_runs.append({
                'seed': s,
                'test_accuracy': float(accuracy_score(yte, pt)),
                'test_macro_f1': float(f1_score(yte, pt, average='macro')),
                'per_class_f1': {c: float(v) for c, v in
                                 zip(CLASSES, f1_score(yte, pt, average=None,
                                                       labels=range(len(CLASSES))))},
                'confusion_matrix': confusion_matrix(yte, pt,
                                                     labels=range(len(CLASSES))).tolist(),
            })
        acc = np.array([r['test_accuracy'] for r in seed_runs])
        mf1 = np.array([r['test_macro_f1'] for r in seed_runs])
        out['models'][name] = {
            'trials': trials, 'selected_hp': best['hp'],
            'val_macro_f1': best['val_macro_f1'],
            'val_spread': float(max(t['val_macro_f1'] for t in trials)
                                - min(t['val_macro_f1'] for t in trials)),
            'test_acc_mean': float(acc.mean()), 'test_acc_sd': float(acc.std(ddof=1)),
            'test_macro_f1_mean': float(mf1.mean()), 'test_macro_f1_sd': float(mf1.std(ddof=1)),
            'seed_runs': seed_runs,
        }
        print(f'  TEST mF1 {mf1.mean():.4f} +- {mf1.std(ddof=1):.4f}  '
              f'acc {acc.mean():.4f} +- {acc.std(ddof=1):.4f}', flush=True)
        print(classification_report(yte, np.array(
            fit_predict(name, best['hp'], Xtr, ytr, Xte, seed=42)),
            target_names=CLASSES, digits=3, zero_division=0))

    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f'\nwritten: {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
