import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import gin

# Project imports
from eval.common import (
    Shapes3DIndexSubset,
    load_model,
    iter_images_from_indices,
    encode_batch_to_latents,
)

# Add EncDiff repo to path for metrics
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ENCDIFF_ROOT = os.path.join(PROJECT_ROOT, 'repos', 'EncDiff')
if ENCDIFF_ROOT not in sys.path:
    sys.path.insert(0, ENCDIFF_ROOT)

from evaluation.metrics import utils as metrics_utils
from evaluation.metrics import dci as dci_mod
from evaluation.metrics import mig as mig_mod
from evaluation.metrics import factor_vae as fvae_mod


def parse_model_spec(spec: str) -> Dict[str, str]:
    # spec format: name|type|config|ckpt
    parts = spec.split('|')
    if len(parts) != 4:
        raise ValueError(f"--model spec must be 'name|type|config|ckpt', got: {spec}")
    name, mtype, config, ckpt = parts
    if mtype not in ('encdiff', 'vqvae'):
        raise ValueError(f"Unknown model type '{mtype}' in spec: {spec}")
    return {'name': name, 'type': mtype, 'config': config, 'ckpt': ckpt}


def make_rep_fn(model, dataset_root: str, device: str, batch_size: int, model_type: str):
    def rep(indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices).reshape(-1)
        outs = []
        if model_type == 'encdiff':
            for x in iter_images_from_indices(dataset_root, idx, batch_size=batch_size, device=device):
                z = encode_batch_to_latents(model, x)
                outs.append(z.numpy())
        else:
            # vqvae: use pre-quantization encoder features, flattened
            import torch
            with torch.no_grad():
                for x in iter_images_from_indices(dataset_root, idx, batch_size=batch_size, device=device):
                    if hasattr(model, 'encode_to_prequant'):
                        h = model.encode_to_prequant(x)
                    else:
                        enc_out = model.encode(x)
                        if isinstance(enc_out, tuple):
                            h = enc_out[0]
                        elif hasattr(enc_out, 'mode'):
                            h = enc_out.mode()
                        else:
                            raise RuntimeError('Unsupported VQ-VAE model encode output')
                    if h.dim() > 2:
                        h = h.reshape(h.shape[0], -1)
                    outs.append(h.float().detach().cpu().numpy())
        Z = np.concatenate(outs, axis=0) if outs else np.zeros((0,))
        return Z
    return rep


def compute_all_metrics(train_ds: Shapes3DIndexSubset,
                        test_ds: Shapes3DIndexSubset,
                        rep_fn,
                        seed: int,
                        mig_n: int,
                        dci_train_n: int,
                        dci_test_n: int,
                        fvae_var_n: int,
                        fvae_train_n: int,
                        fvae_eval_n: int,
                        which: List[str]) -> Dict[str, float]:
    rng = np.random.RandomState(seed)

    results: Dict[str, float] = {}

    # MIG (test-only)
    if 'MIG' in which:
        with gin.unlock_config():
            gin.bind_parameter("discretizer.num_bins", 20)
            gin.bind_parameter("discretizer.discretizer_fn", metrics_utils._histogram_discretize)
        print(f"  MIG: sampling {min(mig_n, len(test_ds.allowed_indices))} test points ...", flush=True)
        mus_te, ys_te = metrics_utils.generate_batch_factor_code(test_ds, rep_fn, min(mig_n, len(test_ds.allowed_indices)), rng, batch_size=64)
        mig = mig_mod._compute_mig(mus_te, ys_te)
        results['MIG'] = float(mig.get('discrete_mig', np.nan))

    # DCI (train -> test)
    if 'DCI' in which:
        print(f"  DCI: train {min(dci_train_n, len(train_ds.allowed_indices))}, test {min(dci_test_n, len(test_ds.allowed_indices))} ...", flush=True)
        mus_tr, ys_tr = metrics_utils.generate_batch_factor_code(train_ds, rep_fn, min(dci_train_n, len(train_ds.allowed_indices)), rng, batch_size=64)
        mus_ts, ys_ts = metrics_utils.generate_batch_factor_code(test_ds, rep_fn, min(dci_test_n, len(test_ds.allowed_indices)), rng, batch_size=64)
        dci = dci_mod._compute_dci(mus_tr, ys_tr, mus_ts, ys_ts)
        results['DCI_Disent'] = float(dci.get('disentanglement', np.nan))
        results['DCI_Compl'] = float(dci.get('completeness', np.nan))
        results['DCI_Info_train'] = float(dci.get('informativeness_train', np.nan))
        results['DCI_Info_test'] = float(dci.get('informativeness_test', np.nan))

    # FactorVAE (test-only protocol with fixed sizes)
    if 'FVAEScore' in which:
        print(f"  FactorVAE: var {fvae_var_n}, train {fvae_train_n}, eval {fvae_eval_n} ...", flush=True)
        with gin.unlock_config():
            gin.bind_parameter("factor_vae_score.batch_size", 64)
            gin.bind_parameter("factor_vae_score.num_variance_estimate", min(fvae_var_n, len(test_ds.allowed_indices)))
            gin.bind_parameter("factor_vae_score.num_train", min(fvae_train_n, len(test_ds.allowed_indices)))
            gin.bind_parameter("factor_vae_score.num_eval", min(fvae_eval_n, len(test_ds.allowed_indices)))
            gin.bind_parameter("prune_dims.threshold", 0.0)
        fvae = fvae_mod.compute_factor_vae(test_ds, rep_fn, rng)
        results['FVAEScore_eval_acc'] = float(fvae.get('eval_accuracy', np.nan))
        results['FVAEScore_train_acc'] = float(fvae.get('train_accuracy', np.nan))
        results['FVAEScore_num_active'] = int(fvae.get('num_active_dims', 0))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', type=str, required=True)
    ap.add_argument('--split', type=str, required=True)
    ap.add_argument('--out', type=str, default='eval/results/final_metrics.csv')
    ap.add_argument('--model', type=str, action='append', required=True,
                   help="Repeat: name|type|config|ckpt where type in {encdiff,vqvae}")
    ap.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--metrics', type=str, nargs='+', default=['MIG','DCI','FVAEScore'],
                    choices=['MIG','DCI','FVAEScore'], help='Which metrics to compute')
    ap.add_argument('--mig-n', type=int, default=10000)
    ap.add_argument('--dci-train-n', type=int, default=10000)
    ap.add_argument('--dci-test-n', type=int, default=5000)
    ap.add_argument('--fvae-var-n', type=int, default=10000)
    ap.add_argument('--fvae-train-n', type=int, default=10000)
    ap.add_argument('--fvae-eval-n', type=int, default=5000)
    args = ap.parse_args()

    # Normalize output path and ensure directory exists
    if not args.out:
        args.out = 'eval/results/final_metrics.csv'
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sp = np.load(args.split)
    train_ds = Shapes3DIndexSubset(sp['train_idx'])
    test_ds = Shapes3DIndexSubset(sp['test_idx'])

    rows = []
    for ms in args.model:
        spec = parse_model_spec(ms)
        name = spec['name']
        print(f"\n==> Model {name} [{spec['type']}]", flush=True)
        t0 = time.time()
        model = load_model(spec['config'], spec['ckpt'], device=args.device)
        rep_fn = make_rep_fn(model, args.dataset_root, args.device, args.batch_size, spec['type'])
        metrics = compute_all_metrics(train_ds, test_ds, rep_fn, args.seed,
                                      args.mig_n, args.dci_train_n, args.dci_test_n,
                                      args.fvae_var_n, args.fvae_train_n, args.fvae_eval_n,
                                      args.metrics)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed/60.0:.1f} min", flush=True)
        metric_row = {'model': name, **metrics}
        rows.append(metric_row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    # Save a small record file
    meta_path = os.path.splitext(args.out)[0] + '_meta.json'
    import json
    meta = {
        'split': args.split,
        'dataset_root': args.dataset_root,
        'seed': args.seed,
        'sizes': {
            'MIG': args.mig_n,
            'DCI_train': args.dci_train_n,
            'DCI_test': args.dci_test_n,
            'FVAEs_var': args.fvae_var_n,
            'FVAEs_train': args.fvae_train_n,
            'FVAEs_eval': args.fvae_eval_n,
        },
        'batch_size': args.batch_size,
        'models': [parse_model_spec(m) for m in args.model],
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metrics → {args.out}\nSaved meta → {meta_path}")


if __name__ == '__main__':
    main()
