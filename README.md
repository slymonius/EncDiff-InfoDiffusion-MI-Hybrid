# Hybrid Objective- and Architecture-Based Disentanglement in Diffusion Models

Official implementation of the hybrid approach combining EncDiff's cross-attention architecture with InfoDiffusion's mutual information regularization for improved disentangled representation learning in diffusion models.

## Key Results

- **13.9% improvement** in MIG score over baseline EncDiff
- Optimal MI regularization weight: ζ = 0.3
- Validated on Shapes3D with 80/20 held-out evaluation split
- Training metrics reliably approximate held-out performance (1.02% MIG difference)

## Method Overview

This work presents the first empirical evaluation combining:
- **Architecture-based** disentanglement (EncDiff's cross-attention mechanism)
- **Objective-based** disentanglement (InfoDiffusion's MI regularization)

The hybrid model applies a weighted MI loss term to encourage independence between latent dimensions while maintaining the structured encoder architecture. An ablation study over 5 MI weights (ζ ∈ {0.0, 0.15, 0.3, 0.5, 1.0}) identifies the optimal balance.

## Repository Structure

```
.
├── scripts/                    # Core training and analysis
│   ├── encdiff_with_mi.py     # Hybrid model implementation
│   ├── mi_loss_module.py      # MI regularization module
│   ├── prepare_report_data.py # Data integration pipeline
│   ├── generate_figures.py    # Publication figures
│   └── generate_tables.py     # Results tables
│
├── eval/                       # Held-out evaluation pipeline
│   ├── 01_make_split.py       # Create 80/20 split
│   ├── 03_eval_metrics_online.py  # Compute MIG/DCI/FactorVAE
│   ├── 04_plot_mig_curve.py   # MIG over epochs
│   ├── sbatch/                # Cluster job templates
│   └── results/               # Evaluation CSVs
│
├── configs/                    # Model configurations
│   ├── shapes3d-vq-4-16-encdiff-hybrid-0.0.yaml   # Baseline
│   ├── shapes3d-vq-4-16-encdiff-hybrid-0.15.yaml
│   ├── shapes3d-vq-4-16-encdiff-hybrid-0.3.yaml   # Optimal
│   ├── shapes3d-vq-4-16-encdiff-hybrid-0.5.yaml
│   └── shapes3d-vq-4-16-encdiff-hybrid-1.0.yaml
│
├── evaluation/                 # Generated outputs
│   ├── figures/               # Publication-quality figures (PDF/PNG)
│   ├── tables/                # Formatted tables (LaTeX/Markdown/HTML)
│   └── *.csv                  # Combined results
│
└── report_materials/          # Paper drafts
    ├── final_report.tex       # IEEE format
    └── report_draft.md        # Draft version
```

## Installation

### 1. Clone EncDiff dependency

```bash
git clone https://github.com/mlpc-ucsd/EncDiff.git repos/EncDiff
```

### 2. Create environment

```bash
conda env create -f environment.yaml
conda activate hybrid-disentangle
```

Or with pip:
```bash
pip install -r requirements.txt
```

### 3. Download Shapes3D dataset

```bash
mkdir -p data/shapes3d
# Download 3dshapes.h5 from https://github.com/deepmind/3d-shapes
wget https://storage.googleapis.com/3d-shapes/3dshapes.h5 -P data/shapes3d/
```

### 4. Obtain pre-trained VQ-VAE

The model requires a pre-trained VQ-VAE checkpoint for Shapes3D. You can either:
- Train one following EncDiff instructions
- Or contact for the checkpoint used in the paper

Place it at: `models/vqvae/shapes3d_vqvae.ckpt`

## Usage

### Training

Train the hybrid model with MI weight ζ=0.3:

```bash
python scripts/encdiff_with_mi.py \
    --config configs/shapes3d-vq-4-16-encdiff-hybrid-0.3.yaml \
    --data_root data/shapes3d
```

For ablation study, train with different ζ values by using the corresponding config files.

### Evaluation

#### 1. Create held-out split

```bash
python eval/01_make_split.py \
    --dataset-root data/shapes3d \
    --out splits/split_v1.npz \
    --seed 42 \
    --train-frac 0.8
```

#### 2. Compute metrics

```bash
python eval/03_eval_metrics_online.py \
    --config configs/shapes3d-vq-4-16-encdiff-hybrid-0.3.yaml \
    --ckpt path/to/checkpoint.ckpt \
    --split splits/split_v1.npz \
    --out eval/results/metrics.csv
```

#### 3. Generate figures and tables

```bash
# Combine data from all sources
python scripts/prepare_report_data.py

# Generate publication figures
python scripts/generate_figures.py

# Generate formatted tables
python scripts/generate_tables.py
```

## Key Files

### Core Implementation

- **`scripts/encdiff_with_mi.py`**: Hybrid model class extending EncDiff's `LatentDiffusion` with MI regularization
- **`scripts/mi_loss_module.py`**: MI loss computation using InfoDiffusion's discrete categorical approach

### Evaluation

- **`eval/03_eval_metrics_online.py`**: Computes MIG, DCI, and FactorVAE on held-out split without pre-exporting latents
- **`eval/results/mig_all_models.csv`**: MIG scores for all models at epochs 15 and 25
- **`eval/results/dci_all_models_2k1k.csv`**: DCI components for all models

### Results

- **`evaluation/figures/figure1_ablation_curve.pdf`**: Inverted-U relationship between ζ and MIG
- **`evaluation/tables/table1_training_vs_holdout.tex`**: Validates training metric reliability
- **`evaluation/tables/table2_final_performance.tex`**: Final scores at epoch 15

## Experimental Details

- **Dataset**: Shapes3D (480,000 images, 6 ground-truth factors)
- **Split**: 80% train (384,000), 20% held-out evaluation (96,000)
- **Metrics**: MIG, DCI (Disentanglement/Completeness/Informativeness), FactorVAE
- **Architecture**: VQ-VAE (f=4, codebook=2048) + latent diffusion (T=1000, U-Net)
- **Training**: 15 epochs standard, 25 epochs extended for ζ=0.3
- **Ablation**: 5 MI weights × 15 epochs = 75 training runs

## Results Summary

| Model | ζ | MIG ↑ | DCI Dis. ↑ | FactorVAE ↑ |
|-------|---|-------|------------|-------------|
| Baseline | 0.0 | 0.361 | 0.993 | 0.823 |
| Hybrid | 0.15 | 0.399 | 0.996 | 0.840 |
| **Hybrid (optimal)** | **0.3** | **0.411** | **0.997** | **0.856** |
| Hybrid | 0.5 | 0.373 | 0.995 | 0.849 |
| Hybrid | 1.0 | 0.254 | 0.990 | 0.779 |

Best values in **bold**. All metrics from held-out evaluation at epoch 15.

## Citation

If you use this code, please cite:

```bibtex
@article{hybrid-disentangle-diffusion,
  title={A Hybrid Objective- and Architecture-Based Approach to Disentangled Representation Learning in Diffusion Models},
  author={[Author Name]},
  year={2025}
}
```

## Acknowledgments

This work builds on:
- **EncDiff**: [Encoder-Based Domain Tuning for Fast Personalization of Text-to-Image Models](https://github.com/mlpc-ucsd/EncDiff)
- **InfoDiffusion**: [Representation Learning Using Diffusion Time](https://arxiv.org/abs/2303.00800)
- **Shapes3D**: [3D Shapes Dataset](https://github.com/deepmind/3d-shapes)

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact [contact info].
