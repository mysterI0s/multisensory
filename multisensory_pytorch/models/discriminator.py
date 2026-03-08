"""
Patch-based spectrogram discriminator for GAN training.

Ported from sourcesep.py make_discrim().
Used optionally during source separation training when gan_weight > 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrogramDiscriminator(nn.Module):
    """
    Patch-based 2D discriminator operating on spectrogram slices.

    Architecture:
        input (B, 1, T, F) → conv1(64, k=4, s=2) → LReLU(0.2) →
        conv2(128, k=4, s=2) → BN → LReLU(0.2) →
        conv3(256, k=4, s=2) → BN → LReLU(0.2) →
        conv4(1, k=4, s=1) → output
    """

    def __init__(self, in_channels=1):
        super().__init__()
        bn_params = dict(momentum=0.0003, eps=1e-5)

        self.conv1 = nn.Conv2d(in_channels, 64, 4, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128, **bn_params)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256, **bn_params)
        self.conv4 = nn.Conv2d(256, 1, 4, stride=1, padding=1, bias=True)

        # Random normal init (TF default for this discriminator)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 1, T, F) spectrogram

        Returns:
            (B, 1, T', F') patch logits
        """
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = self.conv4(x)
        return x
