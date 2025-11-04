"""
EncDiff with MI Loss - Hybrid Model
Combines EncDiff's cross-attention architecture with InfoDiffusion's MI regularisation

Novel contribution: First empirical evaluation combining architectural and
information-theoretic approaches to disentanglement in diffusion models.
"""

import sys
from pathlib import Path

# Add EncDiff repository to path (adjust if needed)
project_root = Path(__file__).parent.parent
encdiff_path = project_root / "repos" / "EncDiff"
if encdiff_path.exists():
    sys.path.insert(0, str(encdiff_path))
else:
    # Fallback: try to find EncDiff in common locations
    import os
    if 'ENCDIFF_PATH' in os.environ:
        sys.path.insert(0, os.environ['ENCDIFF_PATH'])

import torch
import torch.nn as nn
from ldm.models.diffusion.ddpm_enc import LatentDiffusion
from mi_loss_module import MILossModule


class EncDiffWithMI(LatentDiffusion):
    """
    Hybrid model: EncDiff + Mutual Information regularisation

    This extends EncDiff's LatentDiffusion model with an additional
    MI loss term from InfoDiffusion.

    Training objective:
        L_total = L_diffusion + λ_MI * L_MI

    where:
        - L_diffusion: Standard EncDiff diffusion loss
        - L_MI: Mutual information regularisation loss
        - λ_MI: Weight for MI term (default: 0.1 from InfoDiffusion paper)
    """

    def __init__(self, mi_weight=1.0, prior_weight=0.01, *args, **kwargs):
        """
        Args:
            mi_weight: Weight for MI loss term (ζ in InfoDiffusion, default: 1.0 for Shapes3D)
            prior_weight: Weight for prior matching term (λ in InfoDiffusion, default: 0.01 for Shapes3D)
            *args, **kwargs: Arguments for LatentDiffusion
        """
        super().__init__(*args, **kwargs)

        # Initialize MI loss module
        self.mi_loss_module = MILossModule(
            mmd_weight=mi_weight,
            prior_weight=prior_weight,
            kernel='rbf',
            prior_type='gaussian'
        )

        self.mi_weight = mi_weight
        self.prior_weight = prior_weight

        print(f"[Hybrid Model] Initialized EncDiff+MI with MI weight (ζ): {mi_weight}, Prior weight (λ): {prior_weight}")
        print("[Hybrid Model] Training full model (encoder + U-Net) with MI regularization")

    def get_concept_tokens(self, conditioning_input):
        """
        Encode conditioning input into concept tokens suitable for MI loss.

        Args:
            conditioning_input: Tensor (B, C, H, W) or list/tuple containing tensors.

        Returns:
            Tensor of shape (B, latent_dim)
        """
        if isinstance(conditioning_input, (list, tuple)):
            tensors = [item for item in conditioning_input if isinstance(item, torch.Tensor)]
            if not tensors:
                raise ValueError("No tensor-like inputs provided for concept token extraction.")
            conditioning_input = tensors[0]

        if not isinstance(conditioning_input, torch.Tensor):
            raise TypeError(f"Unsupported conditioning input type: {type(conditioning_input)}")

        conditioning_input = conditioning_input.to(self.device)

        concept = self.get_learned_conditioning(conditioning_input)

        if isinstance(concept, (list, tuple)):
            try:
                concept = torch.stack(concept, dim=0)
            except RuntimeError as exc:
                raise RuntimeError(f"Failed to stack conditioning outputs: {exc}")

        if not isinstance(concept, torch.Tensor):
            raise TypeError(f"Conditioning output is not a tensor: {type(concept)}")

        concept = concept.to(self.device)

        if concept.ndim > 2:
            concept = concept.reshape(concept.shape[0], -1)

        return concept

    def training_step(self, batch, batch_idx):
        """
        Training step with both diffusion loss and MI loss.

        Total loss = diffusion_loss + mi_weight * mi_loss
        """
        # Diffusion loss (matches LatentDiffusion training loop)
        x_latent, cond_input = self.get_input(batch, self.first_stage_key)
        diffusion_loss, loss_dict = self(x_latent, cond_input)

        total_loss = diffusion_loss

        try:
            conditioning_images = batch[self.first_stage_key]
            if conditioning_images.ndim == 3:
                conditioning_images = conditioning_images[..., None]
            conditioning_images = conditioning_images.permute(0, 3, 1, 2).contiguous().float().to(self.device)

            concept_tokens = self.get_concept_tokens(conditioning_images)
            mi_results = self.mi_loss_module(concept_tokens, conditioning_images)
            mi_loss = mi_results['mi_loss']

            total_loss = diffusion_loss + mi_loss

            loss_dict['train/loss_diffusion'] = diffusion_loss
            loss_dict['train/mi_loss'] = mi_loss
            loss_dict['train/loss'] = total_loss

        except Exception as exc:
            print(f"[Warning] MI loss computation failed: {exc}")
            loss_dict['train/loss'] = diffusion_loss
            loss_dict['train/loss_diffusion'] = diffusion_loss
            loss_dict['train/mi_loss'] = torch.zeros_like(diffusion_loss)

        self.rec_dict = loss_dict

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step - same as EncDiff (no MI loss in validation)
        """
        return super().validation_step(batch, batch_idx)


def create_hybrid_model(config, mi_weight=0.1):
    """
    Factory function to create hybrid model from config.

    Args:
        config: OmegaConf config dict (from EncDiff yaml)
        mi_weight: Weight for MI loss

    Returns:
        EncDiffWithMI model instance
    """
    from ldm.util import instantiate_from_config

    # Modify config to use our hybrid model
    config.model.target = 'encdiff_with_mi.EncDiffWithMI'
    config.model.params.mi_weight = mi_weight

    # Instantiate model
    model = instantiate_from_config(config.model)

    return model
