import argparse
import glob
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from eval.common import Shapes3DIndexSubset

import sys
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir))
ENCDIFF_ROOT = _os.path.join(PROJECT_ROOT, 'repos', 'EncDiff')
if ENCDIFF_ROOT not in sys.path:
    sys.path.insert(0, ENCDIFF_ROOT)

from evaluation.metrics import utils as metrics_utils
from evaluation.metrics import mig as mig_mod


def make_representation_fn(z: np.ndarray, indices: np.ndarray):
    index_to_row = {int(i): r for r, i in enumerate(indices)}
    def rep(obs: np.ndarray):
        obs = np.asarray(obs).reshape(-1)
        rows = [index_to_row[int(i)] for i in obs]
        return z[np.array(rows, dtype=np.int64)]
    return rep


def compute_mig(test_ds: Shapes3DIndexSubset, rep_fn, num_test: int = 10000, seed: int = 0) -> float:
    rng = np.random.RandomState(seed)
    mus, ys = metrics_utils.generate_batch_factor_code(test_ds, rep_fn, min(num_test, len(test_ds.allowed_indices)), rng, batch_size=64)
    res = mig_mod._compute_mig(mus, ys)
    return float(res.get('discrete_mig', np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=str, required=True, help='Split npz with train_idx/test_idx')
    ap.add_argument('--latents-pattern', dest='latents_pattern', type=str, required=True, help='Pattern with {EPOCH} placeholder or a glob')
    ap.add_argument('--epochs', type=int, nargs='+', required=False, help='Epoch numbers to substitute into pattern')
    ap.add_argument('--out', type=str, required=True, help='Output figure path')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mig-n', type=int, default=10000)
    ap.add_argument('--one-index-labels', action='store_true', help='Add +1 to epoch labels for presentation')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sp = np.load(args.split)
    test_ds = Shapes3DIndexSubset(sp['test_idx'])

    # Resolve files
    files: List[str]
    labels: List[int]
    if '{EPOCH}' in args.latents_pattern:
        assert args.epochs, '--epochs required when using {EPOCH} pattern'
        files = [args.latents_pattern.replace('{EPOCH}', str(e)) for e in args.epochs]
        labels = args.epochs
    else:
        files = sorted(glob.glob(args.latents_pattern))
        # Try to recover epoch numbers from filenames, fallback to index
        labels = list(range(len(files)))

    points = []
    for lf, lab in zip(files, labels):
        cache = np.load(lf)
        z = cache['z'].astype(np.float32)
        idx = cache['indices'].astype(np.int64)
        rep = make_representation_fn(z, idx)
        score = compute_mig(test_ds, rep, num_test=args.mig_n, seed=args.seed)
        points.append((lab, score))
        print(f"epoch={lab}: MIG={score:.4f}")

    points.sort(key=lambda x: x[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    plt.figure(figsize=(6,4))
    plot_x = [x+1 for x in xs] if args.one_index_labels else xs
    plt.plot(plot_x, ys, marker='o')
    plt.xlabel('Epoch' + (' (1-indexed)' if args.one_index_labels else ''))
    plt.ylabel('MIG (test split)')
    plt.title('MIG over epochs (held-out split)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved figure → {args.out}")


if __name__ == '__main__':
    main()
