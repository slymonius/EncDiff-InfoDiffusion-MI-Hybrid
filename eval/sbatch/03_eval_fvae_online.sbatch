#!/bin/bash
#SBATCH --job-name=eval_fvae
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/data/sjag014/764_project/logs/%x_%j.out
#SBATCH --error=/data/sjag014/764_project/logs/%x_%j.err

set -euo pipefail

export PYTHONPATH="/data/sjag014/764_project/repos/EncDiff:/data/sjag014/764_project/repos/taming-transformers:/data/sjag014/764_project/scripts:${PYTHONPATH:-}"
export TORCH_HOME="/data/sjag014/torch_cache"
source /data/sjag014/764_project/env764/bin/activate
cd /data/sjag014/764_project

python eval/03_eval_metrics_online.py \
  --dataset-root "/data/sjag014/764_project/data/shapes3d" \
  --split "splits/split_v1.npz" \
  --out "eval/results/final_metrics_fvae.csv" \
  --batch-size 2048 \
  --metrics FVAEScore \
  --fvae-var-n 2000 --fvae-train-n 2000 --fvae-eval-n 1000 \
  --model "encdiff_base_e15|encdiff|/data/sjag014/764_project/scripts/shapes3d-vq-4-16-encdiff-764-extended.yaml|/data/sjag014/764_project/outputs/shapes3d/encdiff_extended/2025-10-24T08-02-14_shapes3d-vq-4-16-encdiff-764-extended23/checkpoints/last.ckpt" \
  --model "hybrid_mi0.15_e15|encdiff|/data/sjag014/764_project/scripts/shapes3d-vq-4-16-encdiff-hybrid-0.15.yaml|/data/sjag014/764_project/outputs/shapes3d/hybrid_0.15/2025-10-24T20-11-32_shapes3d-vq-4-16-encdiff-hybrid-0.1523/checkpoints/last.ckpt" \
  --model "hybrid_mi0.3_e15|encdiff|/data/sjag014/764_project/scripts/shapes3d-vq-4-16-encdiff-hybrid-0.3.yaml|/data/sjag014/764_project/outputs/shapes3d/hybrid_0.3/2025-10-24T01-09-01_shapes3d-vq-4-16-encdiff-hybrid-0.323/checkpoints/last.ckpt" \
  --model "hybrid_mi0.3_extend_e25|encdiff|/data/sjag014/764_project/scripts/shapes3d-vq-4-16-encdiff-hybrid-0.3-extend.yaml|/data/sjag014/764_project/outputs/shapes3d/hybrid_0.3_extend/2025-10-24T14-30-40_shapes3d-vq-4-16-encdiff-hybrid-0.3-extend23/checkpoints/last.ckpt" \
  --model "hybrid_mi0.5_e15|encdiff|/data/sjag014/764_project/scripts/shapes3d-vq-4-16-encdiff-hybrid-0.5.yaml|/data/sjag014/764_project/outputs/shapes3d/hybrid_0.5/2025-10-24T03-26-01_shapes3d-vq-4-16-encdiff-hybrid-0.523/checkpoints/last.ckpt" \
  --model "hybrid_mi1.0_e15|encdiff|/data/sjag014/764_project/scripts/shapes3d-vq-4-16-encdiff-hybrid-1.0.yaml|/data/sjag014/764_project/outputs/shapes3d/hybrid_1.0/2025-10-24T05-43-54_shapes3d-vq-4-16-encdiff-hybrid-1.023/checkpoints/last.ckpt" \
  --model "vqvae_baseline|vqvae|/data/sjag014/764_project/repos/EncDiff/configs/autoencoder/shapes3d_vq_4_16.yaml|/data/sjag014/764_project/outputs/shapes3d/vqvae/2025-10-22T06-26-50_shapes3d_vq_4_16_76423/checkpoints/last.ckpt"

echo "Saved FVAE metrics → eval/results/final_metrics_fvae.csv"
