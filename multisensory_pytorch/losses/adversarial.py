"""
Adversarial (GAN) losses.

Port of the GAN loss from sourcesep.py — sigmoid cross-entropy for
real/fake discrimination.
"""

import torch
import torch.nn.functional as F


def sigmoid_loss(logits, target_is_real):
    """
    Sigmoid cross-entropy GAN loss.

    Args:
        logits: discriminator output logits
        target_is_real: if True, target = 1 (real); if False, target = 0 (fake)

    Returns:
        scalar loss
    """
    if target_is_real:
        targets = torch.ones_like(logits)
    else:
        targets = torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, targets)


def gan_generator_loss(disc_fake_logits):
    """Generator loss: wants discriminator to think fakes are real."""
    return sigmoid_loss(disc_fake_logits, target_is_real=True)


def gan_discriminator_loss(disc_real_logits, disc_fake_logits):
    """
    Discriminator loss: real → 1, fake → 0.

    Returns:
        total discriminator loss
    """
    real_loss = sigmoid_loss(disc_real_logits, target_is_real=True)
    fake_loss = sigmoid_loss(disc_fake_logits, target_is_real=False)
    return (real_loss + fake_loss) / 2.0
