import argparse
import os
import numpy as np


def detect_dataset_size(dataset_root: str) -> int:
    # Prefer H5 then NPZ; fall back to 480000
    h5 = os.path.join(dataset_root, '3dshapes.h5')
    npz = os.path.join(dataset_root, '3dshapes.npz')
    if os.path.exists(h5):
        import h5py
        with h5py.File(h5, 'r') as f:
            return int(f['images'].shape[0])
    if os.path.exists(npz):
        data = np.load(npz)
        return int(data['images'].shape[0])
    # Default canonical size
    return 480000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', type=str, default='data/shapes3d', help='Path containing 3dshapes.{h5|npz}')
    ap.add_argument('--out', type=str, required=True, help='Output split file, e.g., splits/split_v1.npz')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train-frac', type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    N = detect_dataset_size(args.dataset_root)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(N)
    n_train = int(round(args.train_frac * N))
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:])

    np.savez(args.out, train_idx=train_idx, test_idx=test_idx, seed=args.seed, train_frac=args.train_frac, N=N)
    print(f"Saved split: {args.out}  (N={N}, train={len(train_idx)}, test={len(test_idx)})")


if __name__ == '__main__':
    main()

