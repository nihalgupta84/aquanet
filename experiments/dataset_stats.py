"""Freeze the dataset composition to JSON so figures do not need the images.

The images are not redistributable (D1 derives from a third-party Kaggle upload). Every
composition figure in the paper therefore reads this file rather than scanning `data/`,
which is what lets `make_figures.py` run in the public repository with no data present.

Usage:
    python experiments/dataset_stats.py --out reports/dataset_stats.json
"""
import sys, json, argparse, hashlib, re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['clean', 'algae', 'debris', 'foam', 'oil', 'turbid', 'uncertain']
SETS = {
    'D1': ROOT / 'data' / 'cleaned_water_dataset',
    'D2': ROOT / 'data' / 'cleaned_scrapper_finetune',
    'D3': ROOT / 'data' / 'cleaned_scrapper_unseen_test',
}
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def class_of(p, root):
    parts = p.relative_to(root).parts
    for c in CLASSES:
        if c in parts:
            return c
    return 'unknown'


def scan(root):
    rows = []
    for p in sorted(root.rglob('*')):
        if not (p.is_file() and p.suffix.lower() in EXT):
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
                fmt = im.format
        except Exception:
            continue
        rel = p.relative_to(root)
        split = rel.parts[0] if rel.parts[0] in ('train', 'val', 'test') else '-'
        rows.append({
            'path': str(p.relative_to(ROOT)), 'split': split, 'label': class_of(p, root),
            'w': w, 'h': h, 'format': fmt, 'bytes': p.stat().st_size,
            # Filename stem minus the trailing index: a coarse proxy for acquisition source.
            'source_prefix': re.sub(r'[_-]?\d+$', '', p.stem),
            'md5': hashlib.md5(p.read_bytes()).hexdigest(),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='reports/dataset_stats.json')
    ap.add_argument('--manifest-dir', default='manifests')
    a = ap.parse_args()

    out = {'_meta': {'classes': CLASSES,
                     'note': ('Resolutions are the on-disk sizes before the 224x224 resize. '
                              'The resize does not preserve aspect ratio, so the spread of '
                              'aspect ratios below is a measure of how much geometric '
                              'distortion the models were trained on.')},
           'sets': {}}

    mdir = ROOT / a.manifest_dir
    mdir.mkdir(parents=True, exist_ok=True)

    for name, root in SETS.items():
        if not root.exists():
            print(f'  {name}: missing at {root}, skipped')
            continue
        rows = scan(root)
        by_split = defaultdict(Counter)
        for r in rows:
            by_split[r['split']][r['label']] += 1
        res = Counter(f"{r['w']}x{r['h']}" for r in rows)
        ar = [r['w'] / r['h'] for r in rows]
        # How many images sit in a resolution group that straddles more than one split?
        groups = defaultdict(set)
        for r in rows:
            groups[(r['w'], r['h'])].add(r['split'])
        straddle = sum(1 for r in rows if len(groups[(r['w'], r['h'])]) > 1)

        out['sets'][name] = {
            'n': len(rows),
            'counts_by_split_label': {k: dict(v) for k, v in by_split.items()},
            'counts_by_label': dict(Counter(r['label'] for r in rows)),
            'formats': dict(Counter(r['format'] for r in rows)),
            'top_resolutions': res.most_common(15),
            'n_distinct_resolutions': len(res),
            'aspect_ratio': {
                'min': min(ar), 'max': max(ar),
                'mean': sum(ar) / len(ar),
                'hist_bins': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0],
                'hist_counts': [sum(1 for x in ar if lo <= x < hi) for lo, hi in
                                zip([0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                                    [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 1e9])],
            },
            'n_distinct_source_prefixes': len(set(r['source_prefix'] for r in rows)),
            'top_source_prefixes': Counter(r['source_prefix'] for r in rows).most_common(15),
            'images_in_multi_split_resolution_group': straddle,
            'megapixels_mean': sum(r['w'] * r['h'] for r in rows) / len(rows) / 1e6,
        }
        # Split manifests: file list + MD5, so splits are reproducible without the images.
        man = [{'path': r['path'], 'split': r['split'], 'label': r['label'],
                'md5': r['md5'], 'w': r['w'], 'h': r['h']} for r in rows]
        (mdir / f'{name}_manifest.json').write_text(json.dumps(man, indent=1))
        print(f'  {name}: {len(rows)} images, {len(res)} distinct resolutions, '
              f'{straddle} in multi-split resolution groups')

    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f'written: {p}\nmanifests: {mdir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
