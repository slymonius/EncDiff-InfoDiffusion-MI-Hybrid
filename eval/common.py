import os
import sys
import math
import argparse
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ENCDIFF_ROOT = os.path.join(PROJECT_ROOT, 'repos', 'EncDiff')
if ENCDIFF_ROOT not in sys.path:
    sys.path.insert(0, ENCDIFF_ROOT)

import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


# Shapes3D factor sizes in canonical order
SHAPES3D_SIZES = [10, 10, 10, 8, 4, 15]


def decode_index_to_factors(idx: int, sizes: List[int] = SHAPES3D_SIZES) -> List[int]:
    f = []
    n = idx
    for s in reversed(sizes):
        f.append(n % s)
        n //= s
    return list(reversed(f))


def decode_indices_to_factors(indices: np.ndarray, sizes: List[int] = SHAPES3D_SIZES) -> np.ndarray:
    return np.array([decode_index_to_factors(int(i), sizes) for i in indices], dtype=np.int64)


class Shapes3DIndexSubset:
    """
    Ground-truth adapter restricted to a fixed set of allowed indices.
    Provides sampling compatible with EncDiff's disentanglement metrics.
    Observations returned are integer indices into the full 480k space.
    """

    def __init__(self, allowed_indices: np.ndarray):
        self.allowed_indices = np.array(sorted(set(int(i) for i in allowed_indices)), dtype=np.int64)
        self.allowed_set = set(int(i) for i in self.allowed_indices)
        # Precompute factor tuples for allowed indices and simple lookups
        self._factors = decode_indices_to_factors(self.allowed_indices)
        self._by_dim_val: List[Dict[int, np.ndarray]] = []
        for dim, size in enumerate(SHAPES3D_SIZES):
            buckets: Dict[int, List[int]] = {v: [] for v in range(size)}
            for idx, fac in zip(self.allowed_indices, self._factors):
                buckets[int(fac[dim])].append(int(idx))
            self._by_dim_val.append({k: np.array(v, dtype=np.int64) for k, v in buckets.items()})

    @property
    def num_factors(self) -> int:
        return len(SHAPES3D_SIZES)

    @property
    def factors_num_values(self) -> List[int]:
        return list(SHAPES3D_SIZES)

    def sample(self, num: int, random_state: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        # Uniform over allowed indices
        sel = random_state.choice(self.allowed_indices, size=num, replace=False if num <= len(self.allowed_indices) else True)
        facs = decode_indices_to_factors(sel)
        return facs, sel

    def sample_factors(self, num: int, random_state: np.random.RandomState) -> np.ndarray:
        sel = random_state.choice(self.allowed_indices, size=num, replace=False if num <= len(self.allowed_indices) else True)
        return decode_indices_to_factors(sel)

    def sample_observations_from_factors(self, factors: np.ndarray, random_state: np.random.RandomState) -> np.ndarray:
        """
        Returns observations (indices) consistent with the provided factors on the fixed dimension.
        If exactly one factor dimension is constant across rows, we enforce that value and
        sample the remaining factors from the allowed subset; otherwise, fall back to uniform.
        """
        factors = np.asarray(factors)
        n = factors.shape[0]

        # Detect fixed dimension (exactly one value identical across all rows)
        fixed_dims = [d for d in range(factors.shape[1]) if np.all(factors[:, d] == factors[0, d])]

        if len(fixed_dims) == 1:
            d_fix = fixed_dims[0]
            v_fix = int(factors[0, d_fix])
            pool = self._by_dim_val[d_fix].get(v_fix, np.empty(0, dtype=np.int64))
            if pool.size == 0:
                # If nothing matches, fallback to uniform allowed indices
                return random_state.choice(self.allowed_indices, size=n, replace=False if n <= len(self.allowed_indices) else True)
            # Sample with replacement if needed
            replace = n > len(pool)
            return random_state.choice(pool, size=n, replace=replace)

        # Fallback: uniform over allowed indices
        return random_state.choice(self.allowed_indices, size=n, replace=False if n <= len(self.allowed_indices) else True)

    def sample_observations(self, num: int, random_state: np.random.RandomState) -> np.ndarray:
        """Return a batch of observations consistent with the adapter contract.
        Observations are indices into the full dataset; the representation
        function consumes these indices to load/encode images on-the-fly.
        """
        replace = num > len(self.allowed_indices)
        return random_state.choice(self.allowed_indices, size=num, replace=replace)


def load_model(config_path: str, ckpt_path: str, device: str = 'cuda'):
    cfg = OmegaConf.load(config_path)
    model = instantiate_from_config(cfg.model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    if device == 'cuda' and torch.cuda.is_available():
        model.cuda()
    return model


def iter_images_from_indices(dataset_root: str, indices: np.ndarray, batch_size: int = 256, device: str = 'cuda'):
    """
    Yields batches of images (torch.FloatTensor) in range [-1, 1], shape [B, 3, 64, 64]
    corresponding to the provided indices.
    Supports either 3dshapes.h5 or 3dshapes.npz under dataset_root.
    """
    import h5py

    h5_path = os.path.join(dataset_root, '3dshapes.h5')
    npz_path = os.path.join(dataset_root, '3dshapes.npz')

    if os.path.exists(h5_path):
        f = h5py.File(h5_path, 'r')
        images_ds = f['images']
        N = images_ds.shape[0]
        close_fn = f.close
        def get_img(i):
            return images_ds[i]
    elif os.path.exists(npz_path):
        data = np.load(npz_path)
        images_np = data['images']
        N = images_np.shape[0]
        close_fn = lambda: None
        def get_img(i):
            return images_np[i]
    else:
        raise FileNotFoundError(f"Expected 3dshapes.h5 or 3dshapes.npz under {dataset_root}")

    try:
        device_is_cuda = (device == 'cuda' and torch.cuda.is_available())
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start+batch_size]
            imgs = []
            for i in batch_idx:
                arr = get_img(int(i))  # HWC uint8
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                # [0,255] -> [0,1] -> [-1,1]
                arr = arr / 255.0
                arr = (arr * 2.0) - 1.0
                arr = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
                imgs.append(arr)
            x = torch.stack(imgs, dim=0)
            if device_is_cuda:
                x = x.cuda(non_blocking=True)
            yield x
    finally:
        close_fn()


def encode_batch_to_latents(model, x: torch.Tensor) -> torch.Tensor:
    """Returns 2D latents [B, d] matching training-time metric representation.
    Prefers cond_stage_model.encoding; falls back to get_learned_conditioning.
    """
    with torch.no_grad():
        if hasattr(model, 'cond_stage_model') and hasattr(model.cond_stage_model, 'encoding'):
            z = model.cond_stage_model.encoding(x)
        elif hasattr(model, 'get_learned_conditioning'):
            z = model.get_learned_conditioning(x)
        else:
            raise RuntimeError('Model does not expose cond_stage_model.encoding or get_learned_conditioning')

        # Flatten if model returns higher-rank features
        if z.dim() > 2:
            z = z.reshape(z.shape[0], -1)
        return z.float().detach().cpu()
