"""
Classification losses.

Port of label_loss from videocls.py — softmax cross-entropy with optional
label smoothing, and sigmoid binary classification for ShiftNet.
"""

import torch
import torch.nn.functional as F


def label_loss(logits, labels, smooth=False, num_classes=None):
    """
    Classification loss with optional label smoothing.

    Args:
        logits: (B, C) class logits
        labels: (B,) integer class labels
        smooth: whether to apply label smoothing
        num_classes: required if smooth=True

    Returns:
        loss: scalar loss
        acc: scalar accuracy (detached, for logging only)
    """
    if smooth:
        assert num_classes is not None
        oh = F.one_hot(labels, num_classes).float()
        p = 0.05
        oh = p * (1.0 / num_classes) + (1.0 - p) * oh
        loss = F.cross_entropy(logits, oh)
    else:
        loss = F.cross_entropy(logits, labels)

    # Accuracy (detached, not backpropagated)
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

    return loss, acc


def sigmoid_classification_loss(logits, labels):
    """
    Binary classification loss for ShiftNet (synced vs shifted).

    Args:
        logits: (B, 1) logits
        labels: (B,) binary labels (0 or 1)

    Returns:
        loss, acc
    """
    loss = F.binary_cross_entropy_with_logits(
        logits.squeeze(1), labels.float()
    )

    with torch.no_grad():
        preds = (logits.squeeze(1) > 0).long()
        acc = (preds == labels).float().mean()

    return loss, acc
