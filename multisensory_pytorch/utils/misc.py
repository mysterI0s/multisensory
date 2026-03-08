"""
Miscellaneous utilities.

Replaces various helper functions from tfutil.py and aolib.util that are used
throughout the training and inference code.
"""

import os
import time
import torch
import numpy as np


def moving_avg(name, x, vals, p=0.99):
    """
    Exponential moving average for loss display.

    Args:
        name: key for the tracking dict
        x: current value
        vals: dict to store running averages
        p: decay factor

    Returns:
        Updated running average
    """
    vals[name] = p * vals.get(name, x) + (1 - p) * x
    return vals[name]


def set_device(gpu=None):
    """
    Set and return the torch device.

    Args:
        gpu: GPU index (int), None or -1 for CPU

    Returns:
        torch.device
    """
    if gpu is None or gpu < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu}")


def make_mod(x, m):
    """Round x down to nearest multiple of m."""
    return (x // m) * m


def find_lr(pr, step):
    """
    Compute learning rate at a given step using step decay.

    lr = base_lr * gamma^floor(step / step_size)
    """
    gamma = getattr(pr, "gamma", 0.1)
    scale = gamma ** (step // pr.step_size)
    return pr.base_lr * scale


class CheckpointManager:
    """
    Simple checkpoint manager equivalent to tf.train.Saver.
    
    Saves and loads model/optimizer state dicts.
    """

    def __init__(self, model, optimizer=None, save_dir="checkpoints",
                 max_to_keep=5):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep
        os.makedirs(save_dir, exist_ok=True)

    def save(self, step, filename=None):
        """Save checkpoint."""
        if filename is None:
            filename = f"net-{step}.pt"
        path = os.path.join(self.save_dir, filename)
        state = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
        }
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(state, path)
        print(f"Checkpoint saved: {path}")
        self._cleanup()
        return path

    def load(self, path=None, restore_opt=True):
        """
        Load checkpoint.

        Args:
            path: path to checkpoint file, or None to find latest
            restore_opt: whether to restore optimizer state

        Returns:
            step number
        """
        if path is None:
            path = self._find_latest()
        if path is None:
            print("No checkpoint found.")
            return 0
        print(f"Restoring from: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if restore_opt and self.optimizer is not None and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt.get("step", 0)

    def _find_latest(self):
        """Find the latest checkpoint in save_dir."""
        files = [f for f in os.listdir(self.save_dir) if f.endswith(".pt")]
        if not files:
            return None
        # Sort by step number
        def get_step(f):
            try:
                return int(f.split("-")[1].split(".")[0])
            except (IndexError, ValueError):
                return -1
        files.sort(key=get_step, reverse=True)
        return os.path.join(self.save_dir, files[0])

    def _cleanup(self):
        """Remove old checkpoints beyond max_to_keep."""
        if self.max_to_keep <= 0:
            return
        files = sorted(
            [f for f in os.listdir(self.save_dir) if f.endswith(".pt")],
            key=lambda f: os.path.getmtime(os.path.join(self.save_dir, f)),
            reverse=True,
        )
        for f in files[self.max_to_keep:]:
            os.remove(os.path.join(self.save_dir, f))


class Timer:
    """Simple wall-clock timer for benchmarking."""

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        return self

    def elapsed(self):
        return time.time() - self.start_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        pass
