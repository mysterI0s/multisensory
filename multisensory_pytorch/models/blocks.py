"""
Residual blocks for 2D and 3D convolutions.

Equivalent to block2() and block3() from shift_net.py.
These are the fundamental building blocks used in both ShiftNet and the sound
feature network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _compute_same_padding(kernel_size, stride, dilation=1):
    """
    Compute padding for 'SAME'-like behavior (symmetric version).
    For stride=1 this matches TF SAME exactly.
    For stride>1, we use explicit F.pad in the forward pass.
    """
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]
    if isinstance(stride, int):
        stride = [stride]
    if isinstance(dilation, int):
        dilation = [dilation] * len(kernel_size)
    padding = []
    for k, d in zip(kernel_size, dilation):
        effective_k = k + (k - 1) * (d - 1)
        padding.append((effective_k - 1) // 2)
    return tuple(padding)


def _needs_explicit_pad(kernel_size, stride):
    """Check if we need explicit asymmetric padding (TF SAME with stride > 1)."""
    if isinstance(stride, int):
        stride = [stride]
    return any(s > 1 for s in stride)


def _pad_same_nd(x, kernel_size, stride, dims):
    """
    Apply TF-style 'SAME' padding (asymmetric) for stride > 1.
    This is needed because PyTorch padding='same' only works with stride=1.
    """
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * dims
    if isinstance(stride, int):
        stride = [stride] * dims

    pad = []
    for k, s in reversed(list(zip(kernel_size, stride))):
        # TF SAME formula
        pad_total = max(k - 1, 0)
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        pad.extend([pad_beg, pad_end])
    return F.pad(x, pad)


# ---------------------------------------------------------------------------
# TF BN defaults: decay=0.9997, epsilon=0.001
# PyTorch: momentum = 1 - decay = 0.0003, eps = 0.001
# ---------------------------------------------------------------------------
_BN_MOMENTUM = 0.0003
_BN_EPS = 0.001


class Block2D(nn.Module):
    """
    2D residual block (equivalent to block2 in shift_net.py).

    Structure:
        shortcut ─────────────────────┐
        x → conv1(stride) → BN+ReLU  │
          → conv2 (no BN, no act)     │
          → + ←────────────────────────┘
          → BN → ReLU → output
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
                 stride=1, bn_momentum=_BN_MOMENTUM, bn_eps=_BN_EPS):
        super().__init__()
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)

        self.needs_shortcut = (self.stride != (1, 1) or in_channels != out_channels)

        # Shortcut
        if self.needs_shortcut:
            if self.stride != (1, 1) and in_channels == out_channels:
                self.shortcut = nn.MaxPool2d(kernel_size=1, stride=self.stride)
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=self.stride, bias=False),
                    nn.BatchNorm2d(out_channels, momentum=bn_momentum, eps=bn_eps),
                )
        else:
            self.shortcut = nn.Identity()

        # Main path
        pad1 = _compute_same_padding(self.kernel_size, (1, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels, self.kernel_size,
                               stride=self.stride, padding=pad1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=bn_momentum, eps=bn_eps)

        pad2 = _compute_same_padding(self.kernel_size, (1, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, self.kernel_size,
                               padding=pad2, bias=True)
        # No BN after conv2 — BN applied after residual addition

        self.bn_out = nn.BatchNorm2d(out_channels, momentum=bn_momentum, eps=bn_eps)

    def forward(self, x):
        shortcut = self.shortcut(x)

        # For stride > 1 with SAME padding, we need explicit padding
        if _needs_explicit_pad(self.kernel_size, self.stride):
            out = _pad_same_nd(x, self.kernel_size, self.stride, dims=2)
            # Reset conv1 padding to 0 — we've already padded
            out = self.conv1(out)
        else:
            out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = shortcut + out
        out = F.relu(self.bn_out(out))
        return out


class Block3D(nn.Module):
    """
    3D residual block (equivalent to block3 in shift_net.py).

    Structure:
        shortcut ─────────────────────────────┐
        x → [bottleneck(1x1x1)] → conv1(stride, rate) → BN+ReLU  │
          → conv2(rate, no BN/act)                                 │
          → + ←────────────────────────────────────────────────────┘
          → BN → ReLU → output  (or just ReLU if use_bn=False)
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 stride=1, rate=1, bottleneck=False, use_bn=True,
                 bn_momentum=_BN_MOMENTUM, bn_eps=_BN_EPS):
        super().__init__()
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        elif len(stride) == 3:
            self.stride = tuple(stride)
        else:
            self.stride = tuple(stride)
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if isinstance(rate, int):
            self.rate = (rate, rate, rate)
        else:
            self.rate = tuple(rate)

        self.use_bn = use_bn
        self.needs_shortcut = (self.stride != (1, 1, 1) or in_channels != out_channels)
        self.bottleneck = bottleneck

        # Shortcut
        if self.needs_shortcut:
            if self.stride != (1, 1, 1) and in_channels == out_channels:
                self.shortcut = nn.MaxPool3d(kernel_size=1, stride=self.stride)
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, 1, stride=self.stride, bias=False),
                    nn.BatchNorm3d(out_channels, momentum=bn_momentum, eps=bn_eps),
                )
        else:
            self.shortcut = nn.Identity()

        # Optional bottleneck
        if bottleneck:
            self.bottleneck_conv = nn.Conv3d(in_channels, out_channels, 1,
                                             dilation=self.rate, bias=False)
            self.bottleneck_bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum, eps=bn_eps)
            conv1_in = out_channels
        else:
            conv1_in = in_channels

        # Main path — conv1
        pad1 = _compute_same_padding(self.kernel_size, (1, 1, 1), self.rate)
        self.conv1 = nn.Conv3d(conv1_in, out_channels, self.kernel_size,
                               stride=self.stride, padding=pad1,
                               dilation=self.rate, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum, eps=bn_eps)

        # Main path — conv2 (no BN, no activation)
        pad2 = _compute_same_padding(self.kernel_size, (1, 1, 1), self.rate)
        self.conv2 = nn.Conv3d(out_channels, out_channels, self.kernel_size,
                               padding=pad2, dilation=self.rate, bias=True)

        # Post-addition BN
        if use_bn:
            self.bn_out = nn.BatchNorm3d(out_channels, momentum=bn_momentum, eps=bn_eps)

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = x
        if self.bottleneck:
            out = F.relu(self.bottleneck_bn(self.bottleneck_conv(out)))

        # For stride > 1 with SAME padding, explicit pad
        if _needs_explicit_pad(self.kernel_size, self.stride):
            out = _pad_same_nd(out, self.kernel_size, self.stride, dims=3)
            out = self.conv1(out)
        else:
            out = self.conv1(out)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)

        out = shortcut + out
        if self.use_bn:
            out = F.relu(self.bn_out(out))
        else:
            out = F.relu(out)
        return out


class Conv3dSame(nn.Module):
    """
    Conv3d with TF-style SAME padding for any stride.
    Wraps nn.Conv3d with explicit asymmetric padding when stride > 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        if isinstance(dilation, int):
            dilation = (dilation,) * 3

        self.kernel_size = kernel_size
        self.stride = stride

        # For stride=1, use standard padding
        if all(s == 1 for s in stride):
            padding = _compute_same_padding(kernel_size, stride, dilation)
        else:
            padding = 0  # We'll pad explicitly in forward

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              bias=bias)
        self.needs_explicit_pad = any(s > 1 for s in stride)

    def forward(self, x):
        if self.needs_explicit_pad:
            x = _pad_same_nd(x, self.kernel_size, self.stride, dims=3)
        return self.conv(x)


class Conv2dSame(nn.Module):
    """
    Conv2d with TF-style SAME padding for any stride.
    Port of conv2d_same from tfutil.py.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(dilation, int):
            dilation = (dilation,) * 2

        self.kernel_size = kernel_size
        self.stride = stride

        if all(s == 1 for s in stride):
            padding = _compute_same_padding(kernel_size, stride, dilation)
        else:
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              bias=bias)
        self.needs_explicit_pad = any(s > 1 for s in stride)

    def forward(self, x):
        if self.needs_explicit_pad:
            x = _pad_same_nd(x, self.kernel_size, self.stride, dims=2)
        return self.conv(x)
