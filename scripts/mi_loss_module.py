"""
Mutual Information Loss Module
Extracted from InfoDiffusion (Wang et al., 2023)
https://github.com/isjakewong/InfoDiffusion

This module implements Maximum Mean Discrepancy (MMD) based
mutual information estimation for use in EncDiff hybrid model.
"""

import torch
import torch.nn as nn
import numpy as np


def compute_kernel(x, y, kernel_type='rbf', bandwidth=None):
    """
    Compute kernel matrix between two sets of samples.

    Args:
        x: Tensor of shape (batch_size, dim)
        y: Tensor of shape (batch_size, dim)
        kernel_type: Type of kernel ('rbf' or 'imq')
        bandwidth: Kernel bandwidth (auto-computed if None)

    Returns:
        Kernel matrix of shape (batch_size, batch_size)
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    # Compute pairwise distances
    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)

    # Euclidean distance squared
    dist_sq = ((tiled_x - tiled_y) ** 2).sum(dim=2)

    # Auto-compute bandwidth if not provided
    if bandwidth is None:
        # Median heuristic
        bandwidth = torch.median(dist_sq[dist_sq > 0])
        bandwidth = torch.sqrt(0.5 * bandwidth / np.log(x_size + 1))

    # Apply kernel
    if kernel_type == 'rbf':
        # Radial Basis Function (Gaussian) kernel
        kernel_val = torch.exp(-dist_sq / (2 * bandwidth ** 2))
    elif kernel_type == 'imq':
        # Inverse Multi-Quadratic kernel
        kernel_val = 1.0 / (1.0 + dist_sq / bandwidth)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return kernel_val


def mmd_loss(source_samples, target_samples, kernel='rbf', bandwidth=None):
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.

    MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    where x ~ P, y ~ Q, and k is a kernel function

    Args:
        source_samples: Samples from first distribution (batch_size, dim)
        target_samples: Samples from second distribution (batch_size, dim)
        kernel: Kernel type ('rbf' or 'imq')
        bandwidth: Kernel bandwidth (auto-computed if None)

    Returns:
        MMD² value (scalar tensor)
    """
    # Compute kernel matrices
    xx = compute_kernel(source_samples, source_samples, kernel, bandwidth)
    yy = compute_kernel(target_samples, target_samples, kernel, bandwidth)
    xy = compute_kernel(source_samples, target_samples, kernel, bandwidth)

    # Compute MMD²
    # E[k(x,x')] - diagonal (don't compare sample with itself)
    xx_mean = (xx.sum() - xx.diag().sum()) / (xx.size(0) * (xx.size(0) - 1))

    # E[k(y,y')] - diagonal
    yy_mean = (yy.sum() - yy.diag().sum()) / (yy.size(0) * (yy.size(0) - 1))

    # E[k(x,y)]
    xy_mean = xy.mean()

    # MMD²
    mmd_sq = xx_mean + yy_mean - 2 * xy_mean

    return mmd_sq


def compute_mi_loss(encoder_outputs, images, prior_samples,
                    mmd_weight=1.0, prior_weight=0.01, kernel='rbf'):
    """
    Compute mutual information loss using MMD.

    This estimates MI(x; z) by computing:
    MMD²(p(x,z) || p(x)p(z))

    Where:
    - p(x,z) is the joint distribution (actual encoder outputs)
    - p(x)p(z) is the product of marginals (shuffled pairs)

    Args:
        encoder_outputs: Latent codes from encoder (batch_size, latent_dim)
        images: Input images (for joint distribution)
        prior_samples: Samples from prior p(z) (batch_size, latent_dim)
        mmd_weight: Weight for MMD loss (ζ in InfoDiffusion paper)
        prior_weight: Weight for prior matching loss (λ in InfoDiffusion paper)
        kernel: Kernel type for MMD

    Returns:
        mi_loss: Mutual information loss term (scalar)
    """
    batch_size = encoder_outputs.size(0)

    # Joint distribution: actual (image, latent) pairs
    joint_samples = encoder_outputs

    # Product of marginals: shuffle latents to break dependencies
    # This approximates sampling from p(x)p(z)
    indices = torch.randperm(batch_size, device=encoder_outputs.device)
    marginal_samples = encoder_outputs[indices]

    # Compute MMD between joint and marginal
    mmd_value = mmd_loss(joint_samples, marginal_samples, kernel=kernel)

    # Also compute MMD to prior (encourages latents to match prior distribution)
    if prior_samples is not None:
        mmd_to_prior = mmd_loss(encoder_outputs, prior_samples, kernel=kernel)
    else:
        mmd_to_prior = 0.0

    # Total MI loss
    # Maximize MI = minimize negative MI
    # We want high MI, so we minimize -MMD (or maximize MMD)
    # But we also want latents close to prior, so we add MMD to prior
    mi_loss = -mmd_weight * mmd_value + prior_weight * mmd_to_prior

    return mi_loss


class MILossModule(nn.Module):
    """
    Module wrapper for MI loss computation.
    Can be integrated into EncDiff training.
    """
    def __init__(self, mmd_weight=1.0, prior_weight=0.01, kernel='rbf', prior_type='gaussian'):
        super().__init__()
        self.mmd_weight = mmd_weight
        self.prior_weight = prior_weight
        self.kernel = kernel
        self.prior_type = prior_type

    def sample_prior(self, batch_size, latent_dim, device):
        """Sample from prior distribution p(z)"""
        if self.prior_type == 'gaussian':
            return torch.randn(batch_size, latent_dim, device=device)
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")

    def forward(self, encoder_outputs, images):
        """
        Compute MI loss.

        Args:
            encoder_outputs: (batch_size, latent_dim)
            images: (batch_size, channels, height, width) - not used but kept for compatibility

        Returns:
            Loss dict with 'mi_loss' key
        """
        batch_size, latent_dim = encoder_outputs.shape

        # Sample from prior
        prior_samples = self.sample_prior(batch_size, latent_dim, encoder_outputs.device)

        # Compute MI loss
        mi_loss = compute_mi_loss(
            encoder_outputs,
            images,
            prior_samples,
            mmd_weight=self.mmd_weight,
            prior_weight=self.prior_weight,
            kernel=self.kernel
        )

        return {
            'mi_loss': mi_loss,
            'mmd_weight': self.mmd_weight
        }
