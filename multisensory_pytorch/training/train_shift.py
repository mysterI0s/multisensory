"""
Training script for ShiftNet (audio-visual correspondence).

Port of shift_net.py Model.train() — PyTorch training loop with
momentum optimizer, gradient clipping, and binary classification.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..models.shift_net import ShiftNet
from ..losses.classification import sigmoid_classification_loss
from ..datasets.shift_dataset import ShiftDataset
from ..utils.misc import moving_avg, CheckpointManager, set_device
from ..utils.params import shift_v1


def train_shift(pr, device='cuda:0', restore=False, restore_opt=True):
    """Main training loop for ShiftNet."""
    device = set_device(0 if 'cuda' in device else -1)
    print(pr)

    # Model
    model = ShiftNet(pr, use_sound=pr.use_sound).to(device)

    # Optimizer
    if pr.opt_method == 'momentum':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=pr.base_lr,
            momentum=pr.momentum_rate, weight_decay=pr.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=pr.base_lr,
            weight_decay=pr.weight_decay
        )

    # LR scheduler: step decay
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=getattr(pr, 'step_size', 50000),
        gamma=getattr(pr, 'gamma', 0.1)
    )

    # Dataset
    dataset = ShiftDataset(pr.train_list, pr, train=True)
    dataloader = DataLoader(
        dataset,
        batch_size=pr.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Checkpointing
    ckpt = CheckpointManager(model, optimizer,
                              save_dir=pr.train_dir, max_to_keep=5)

    step = 0
    if restore:
        step = ckpt.load(restore_opt=restore_opt)

    # TensorBoard
    writer = SummaryWriter(log_dir=pr.summary_dir)

    # AMP
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device='cuda') if use_amp else None

    # L2 regularization through weight_decay in optimizer (already set above)

    model.train()
    val_hist = {}
    data_iter = iter(dataloader)

    print(f"Starting ShiftNet training from step {step}")
    while step < pr.train_iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        ims = batch['ims'].to(device)         # (B, 3, T, H, W)
        samples = batch['samples'].to(device)  # (B, N, 2)
        labels = batch['label'].to(device)     # (B,)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, cam, last_conv, im_net, scales, im_scales = model(ims, samples)
            loss, acc = sigmoid_classification_loss(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), pr.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), pr.grad_clip)
            optimizer.step()

        scheduler.step()
        step += 1

        # Logging
        if step % pr.print_iters == 0 or step < 10:
            lr = optimizer.param_groups[0]['lr']
            l = moving_avg('loss', loss.item(), val_hist)
            a = moving_avg('acc', acc.item(), val_hist)
            print(f"Step {step}, lr={lr:.1e}, loss: {l:.4f}, acc: {a:.4f}")

        if step % pr.summary_iters == 0:
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("acc/train", acc.item(), step)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], step)

        if step > 0 and step % pr.check_iters == 0:
            ckpt.save(step)

    print("ShiftNet training complete.")
    ckpt.save(step)
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train ShiftNet")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    pr = shift_v1()
    if args.batch_size:
        pr.batch_size = args.batch_size

    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    train_shift(pr, device=device, restore=args.restore)


if __name__ == "__main__":
    main()
