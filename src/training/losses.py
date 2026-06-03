"""
Loss Functions for MeshVTON Training.

Composite loss: L1, perceptual (VGG), LPIPS, adversarial, and KL divergence.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TryOnLoss(nn.Module):
    """
    Composite loss for virtual try-on training.

    Args:
        l1_weight: Weight for L1 reconstruction loss.
        perceptual_weight: Weight for VGG perceptual loss.
        lpips_weight: Weight for LPIPS perceptual loss.
        adversarial_weight: Weight for adversarial loss.
        kl_weight: Weight for KL divergence loss (VAE).
    """

    def __init__(self, l1_weight=1.0, perceptual_weight=0.5, lpips_weight=1.0,
                 adversarial_weight=0.1, kl_weight=0.0001):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.lpips_weight = lpips_weight
        self.adversarial_weight = adversarial_weight
        self.kl_weight = kl_weight

        if perceptual_weight > 0:
            self.perceptual = PerceptualLoss()
        if lpips_weight > 0:
            self.lpips = LPIPSLoss()
        if adversarial_weight > 0:
            self.discriminator = PatchDiscriminator()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mean: Optional[torch.Tensor] = None,
                logvar: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        losses = {}
        total = torch.tensor(0.0, device=pred.device)

        # L1 reconstruction
        if self.l1_weight > 0:
            l1 = F.l1_loss(pred, target)
            losses["l1"] = l1
            total = total + self.l1_weight * l1

        # Perceptual (VGG)
        if self.perceptual_weight > 0:
            perc = self.perceptual(pred, target)
            losses["perceptual"] = perc
            total = total + self.perceptual_weight * perc

        # LPIPS
        if self.lpips_weight > 0:
            lpips_val = self.lpips(pred, target)
            losses["lpips"] = lpips_val
            total = total + self.lpips_weight * lpips_val

        # Adversarial
        if self.adversarial_weight > 0:
            adv = self._adversarial_loss(pred, target)
            losses["adversarial"] = adv
            total = total + self.adversarial_weight * adv

        # KL divergence
        if self.kl_weight > 0 and mean is not None and logvar is not None:
            kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            losses["kl"] = kl
            total = total + self.kl_weight * kl

        losses["total"] = total
        return losses

    def _adversarial_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fake_score = self.discriminator(pred)
        return F.binary_cross_entropy_with_logits(
            fake_score, torch.ones_like(fake_score)
        )


class PerceptualLoss(nn.Module):
    """VGG-19 perceptual loss comparing intermediate feature activations."""

    def __init__(self, layer_weights=None):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.blocks = nn.ModuleList([
            vgg[:4], vgg[4:9], vgg[9:18], vgg[18:27], vgg[27:36],
        ])
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        self.weights = layer_weights or [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = torch.tensor(0.0, device=pred.device)
        x, y = pred, target
        for block, w in zip(self.blocks, self.weights):
            x, y = block(x), block(y)
            loss = loss + w * F.l1_loss(x, y)
        return loss


class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss wrapper."""

    def __init__(self):
        super().__init__()
        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net="alex", verbose=False)
            for param in self.loss_fn.parameters():
                param.requires_grad = False
        except ImportError:
            self.loss_fn = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_fn is not None:
            return self.loss_fn(pred, target).mean()
        return F.l1_loss(pred, target)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for adversarial training."""

    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2)]
        ch = ndf
        for i in range(1, n_layers):
            out_ch = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, out_ch, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2),
            ]
            ch = out_ch
        layers.append(nn.Conv2d(ch, 1, 4, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
