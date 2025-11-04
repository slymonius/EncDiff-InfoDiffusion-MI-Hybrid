Evaluation pipeline for held-out Shapes3D metrics

Overview
- Creates a fixed 80/20 split of Shapes3D indices.
- Exports latents z for selected indices per model checkpoint.
- Computes MIG (test-only), DCI (train→test), and FactorVAE (test-only) on the fixed split.
- Plots a MIG-over-epochs curve for a chosen model (e.g., MI=0.3).

Key scripts
- eval/01_make_split.py – generate split file with train_idx/test_idx
- eval/02_export_latents.py – encode selected indices to latents and cache to NPZ
- eval/03_eval_metrics.py – compute metrics from cached latents using the split
- eval/04_plot_mig_curve.py – recompute MIG per epoch on the fixed test split and plot

Conventions
- Outputs under splits/, eval/cache/, eval/results/, figs/ (not in prd03_copy/).
- Uses repos/EncDiff/evaluation/metrics implementation; sampling is restricted to the 80/20 split via a custom Shapes3D subset wrapper.

Quick examples
- Make a split: `python eval/01_make_split.py --dataset-root data/shapes3d --out splits/split_v1.npz --seed 42 --train-frac 0.8`
- Export latents: `python eval/02_export_latents.py --dataset-root data/shapes3d --config configs/shapes3d-vq-4-16-encdiff-hybrid-0.3.yaml --ckpt /path/to/checkpoint.ckpt --model-name hybrid_mi0.3_e25 --split splits/split_v1.npz --subset all --out eval/cache/hybrid_mi0.3_e25_latents.npz`
- Compute metrics: `python eval/03_eval_metrics.py --split splits/split_v1.npz --latents eval/cache/*.npz --out eval/results/final_metrics.csv`
- MIG curve: `python eval/04_plot_mig_curve.py --split splits/split_v1.npz --latents-pattern "eval/cache/hybrid_mi0.3_e{EPOCH}_latents.npz" --epochs 5 10 15 20 25 --out figs/mig_curve_mi0.3.png`

Notes
- Latent export computes the same representation used in training-time metrics: `cond_stage_model.encoding(x)` when available; otherwise it falls back to `get_learned_conditioning(x)`.
- The FactorVAE probe sampling is constrained to the test split by generating batches whose fixed factor value matches the requested one while sampling the other factors from the allowed subset. This preserves the protocol while ensuring evaluation-only sampling.

