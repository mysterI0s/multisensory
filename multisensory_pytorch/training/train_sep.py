"""
Training script for source separation.

Port of sourcesep.py Model.train() — PyTorch training loop with:
- Mixed precision (AMP)
- Gradient clipping
- Periodic checkpointing (fast + slow)
- TensorBoard logging
- Optional DDP multi-GPU
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..models.sourcesep import SourceSepUNet
from ..models.shift_net import ShiftNet
from ..models.discriminator import SpectrogramDiscriminator
from ..losses.separation import SeparationLoss, PITLoss
from ..losses.adversarial import gan_generator_loss, gan_discriminator_loss
from ..datasets.sep_dataset import SeparationDataset
from ..utils.audio import stft, normalize_rms
from ..utils.misc import moving_avg, CheckpointManager, set_device
from ..utils.params import sep_full, sep_unet_pit


def train_separation(pr, device='cuda:0', restore=False, restore_opt=True):
    """
    Main training function for source separation.

    Args:
        pr: Params object (from sep_full() or sep_unet_pit())
        device: torch device string
        restore: whether to restore from latest checkpoint
        restore_opt: whether to restore optimizer state
    """
    device = set_device(0 if device == 'cuda:0' else -1)
    print(pr)

    # Build ShiftNet (frozen) for video features
    shift_net = None
    if pr.net_style != 'no-im':
        shift_net = ShiftNet(pr).to(device)
        shift_net.eval()
        for p in shift_net.parameters():
            p.requires_grad = False
        print("ShiftNet loaded (frozen)")

    # Build model
    model = SourceSepUNet(pr, shift_net=shift_net, net_style=pr.net_style).to(device)

    # Optimizer
    gen_params = [p for p in model.parameters() if p.requires_grad]
    if pr.opt_method == 'adam':
        optimizer = torch.optim.Adam(gen_params, lr=pr.base_lr,
                                     weight_decay=pr.weight_decay)
    else:
        optimizer = torch.optim.SGD(gen_params, lr=pr.base_lr,
                                    momentum=0.9, weight_decay=pr.weight_decay)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=pr.step_size, gamma=pr.gamma
    )

    # Discriminator (optional)
    discriminator = None
    disc_optimizer = None
    if pr.gan_weight > 0:
        discriminator = SpectrogramDiscriminator().to(device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(),
                                          lr=pr.base_lr)

    # Loss
    sep_loss_fn = SeparationLoss(pr)
    pit_loss_fn = PITLoss(pr) if pr.pit_weight > 0 else None

    # Dataset + DataLoader
    dataset = SeparationDataset(pr.train_list, pr, train=True)
    dataloader = DataLoader(
        dataset,
        batch_size=pr.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    # Checkpointing
    ckpt_fast = CheckpointManager(model, optimizer,
                                   save_dir=os.path.join(pr.train_dir),
                                   max_to_keep=5)
    ckpt_slow = CheckpointManager(model, optimizer,
                                   save_dir=os.path.join(pr.train_dir, 'slow'),
                                   max_to_keep=1000)

    # Restore
    step = 0
    if restore:
        step = ckpt_fast.load(restore_opt=restore_opt)

    # TensorBoard
    writer = SummaryWriter(log_dir=pr.summary_dir)

    # Mixed precision
    scaler = torch.amp.GradScaler(device='cuda') if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'

    # Training loop
    model.train()
    val_hist = {}
    data_iter = iter(dataloader)

    print(f"Starting training from step {step}")
    while step < pr.train_iters:
        # Get batch (restart iterator if exhausted)
        try:
            batch = next(data_iter)
        except (StopIteration, RuntimeError):
            data_iter = iter(dataloader)
            batch = next(data_iter)

        ims = batch['ims'].to(device)
        samples_mix = batch['samples_mix'].to(device)
        samples_fg = batch['samples_fg'].to(device)
        samples_bg = batch['samples_bg'].to(device)

        # Compute spectrograms
        with torch.no_grad():
            spec_mix, phase_mix = stft(samples_mix[:, :, 0], pr)
            spec_mix = spec_mix[:, :pr.spec_len]
            phase_mix = phase_mix[:, :pr.spec_len]

            spec_fg, phase_fg = stft(samples_fg[:, :, 0], pr)
            spec_fg = spec_fg[:, :pr.spec_len]
            phase_fg = phase_fg[:, :pr.spec_len]

            spec_bg, phase_bg = stft(samples_bg[:, :, 0], pr)
            spec_bg = spec_bg[:, :pr.spec_len]
            phase_bg = phase_bg[:, :pr.spec_len]

        # Forward pass (with optional AMP)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            results = model(ims, samples_mix, spec_mix, phase_mix)
            pred_spec_fg, pred_wav_fg, pred_phase_fg = results[0], results[1], results[2]
            pred_spec_bg, pred_wav_bg, pred_phase_bg = results[3], results[4], results[5]

            # Compute loss
            total_loss = torch.tensor(0.0, device=device)
            loss_dict = {}

            if 'fg-bg' in pr.loss_types:
                gen_loss, gen_losses = sep_loss_fn(
                    pred_spec_fg, pred_spec_bg, pred_phase_fg, pred_phase_bg,
                    spec_fg, spec_bg, phase_fg, phase_bg
                )
                total_loss = total_loss + gen_loss
                loss_dict.update(gen_losses)

            if 'pit' in pr.loss_types and pit_loss_fn is not None:
                pit_l = pit_loss_fn(pred_spec_fg, pred_spec_bg, spec_fg, spec_bg)
                total_loss = total_loss + pr.pit_weight * pit_l
                loss_dict['pit'] = pit_l

        # Weight decay is in optimizer, so just the loss
        # (TF adds slim regularization losses separately — we use optimizer weight_decay instead)

        # Backward
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(gen_params, pr.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, pr.grad_clip)
            optimizer.step()

        scheduler.step()
        step += 1

        # GAN training step
        if discriminator is not None and pr.gan_weight > 0:
            disc_optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                real_logits = discriminator(spec_fg.unsqueeze(1))
                fake_logits = discriminator(pred_spec_fg.detach().unsqueeze(1))
                d_loss = gan_discriminator_loss(real_logits, fake_logits)

            if scaler:
                scaler.scale(d_loss).backward()
                scaler.step(disc_optimizer)
                scaler.update()
            else:
                d_loss.backward()
                disc_optimizer.step()
            loss_dict['disc'] = d_loss

        # Logging
        if step % pr.print_iters == 0 or step < 10:
            lr = optimizer.param_groups[0]['lr']
            out_parts = []
            for name, val in loss_dict.items():
                v = val.item() if isinstance(val, torch.Tensor) else val
                out_parts.append(f"{name}: {moving_avg(name, v, val_hist):.4f}")
            print(f"Step {step}, lr={lr:.1e}, {' '.join(out_parts)}")

        # TensorBoard
        if step % pr.summary_iters == 0:
            for name, val in loss_dict.items():
                v = val.item() if isinstance(val, torch.Tensor) else val
                writer.add_scalar(f"loss/{name}", v, step)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], step)

        # Checkpoints
        if step > 0 and step % pr.check_iters == 0:
            ckpt_fast.save(step)
        if step > 0 and step % pr.slow_check_iters == 0:
            ckpt_slow.save(step)

    print("Training complete.")
    ckpt_fast.save(step)
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train source separation model")
    parser.add_argument("--model", type=str, default="full",
                        choices=["full", "unet-pit"],
                        help="Model variant")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (-1 for CPU)")
    parser.add_argument("--restore", action="store_true", help="Restore from checkpoint")
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    if args.model == "full":
        pr = sep_full()
    elif args.model == "unet-pit":
        pr = sep_unet_pit()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.batch_size is not None:
        pr.batch_size = args.batch_size

    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    train_separation(pr, device=device, restore=args.restore)


if __name__ == "__main__":
    main()
