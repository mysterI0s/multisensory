"""
Source separation losses.

Port of the loss functions from sourcesep.py: L1 spec loss, phase loss,
Permutation Invariant Training (PIT) loss, and combined separation loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.audio import normalize_spec, normalize_phase


class SeparationLoss(nn.Module):
    """
    Combined loss for source separation.

    Components:
        - L1 on normalized spectrogram (magnitude)
        - L1 on normalized phase (weighted by phase_weight)
        - Optional PIT loss
        - Optional GAN loss (handled separately)
    """

    def __init__(self, pr):
        super().__init__()
        self.pr = pr
        self.l1_weight = pr.l1_weight
        self.phase_weight = pr.phase_weight

    def _spec_loss(self, pred, target, pr):
        """L1 loss on normalized spectrogram magnitude."""
        return torch.mean(torch.abs(normalize_spec(pred, pr) - normalize_spec(target, pr)))

    def _phase_loss(self, pred, target, pr):
        """L1 loss on normalized phase."""
        return torch.mean(torch.abs(normalize_phase(pred) - normalize_phase(target)))

    def forward(self, pred_spec_fg, pred_spec_bg, pred_phase_fg, pred_phase_bg,
                gt_spec_fg, gt_spec_bg, gt_phase_fg, gt_phase_bg):
        """
        Compute separation loss.

        Args:
            pred_spec_fg, pred_spec_bg: predicted spectrogram magnitudes
            pred_phase_fg, pred_phase_bg: predicted phases
            gt_spec_fg, gt_spec_bg: ground truth spectrogram magnitudes
            gt_phase_fg, gt_phase_bg: ground truth phases

        Returns:
            total_loss, loss_dict with individual components
        """
        pr = self.pr
        losses = {}

        # Foreground
        fg_spec_loss = self._spec_loss(pred_spec_fg, gt_spec_fg, pr)
        fg_phase_loss = self._phase_loss(pred_phase_fg, gt_phase_fg, pr)
        losses['fg_spec'] = fg_spec_loss
        losses['fg_phase'] = fg_phase_loss

        # Background
        bg_spec_loss = self._spec_loss(pred_spec_bg, gt_spec_bg, pr)
        bg_phase_loss = self._phase_loss(pred_phase_bg, gt_phase_bg, pr)
        losses['bg_spec'] = bg_spec_loss
        losses['bg_phase'] = bg_phase_loss

        # Combined
        total = self.l1_weight * (fg_spec_loss + bg_spec_loss) + \
                self.phase_weight * (fg_phase_loss + bg_phase_loss)

        losses['total'] = total
        return total, losses


class PITLoss(nn.Module):
    """
    Permutation Invariant Training (PIT) loss.

    Computes loss for both assignments (fg→src0/bg→src1 and fg→src1/bg→src0)
    and uses the lower loss. This allows the model to learn without knowing
    which source is "foreground" vs "background".
    """

    def __init__(self, pr):
        super().__init__()
        self.pr = pr

    def _pair_loss(self, pred_spec_a, pred_spec_b,
                   gt_spec_0, gt_spec_1, pr):
        """L1 loss for one assignment."""
        loss_a = torch.mean(torch.abs(
            normalize_spec(pred_spec_a, pr) - normalize_spec(gt_spec_0, pr)))
        loss_b = torch.mean(torch.abs(
            normalize_spec(pred_spec_b, pr) - normalize_spec(gt_spec_1, pr)))
        return loss_a + loss_b

    def forward(self, pred_spec_fg, pred_spec_bg,
                gt_spec_0, gt_spec_1):
        """
        Compute PIT loss.

        Returns:
            min_loss: the minimum over both permutations
        """
        pr = self.pr
        # Assignment 1: fg→0, bg→1
        loss1 = self._pair_loss(pred_spec_fg, pred_spec_bg,
                                gt_spec_0, gt_spec_1, pr)
        # Assignment 2: fg→1, bg→0
        loss2 = self._pair_loss(pred_spec_fg, pred_spec_bg,
                                gt_spec_1, gt_spec_0, pr)

        return torch.min(loss1, loss2)
