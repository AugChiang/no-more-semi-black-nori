import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import RestorationDataset
from model import DualDomainNAFNet


class CharbonnierLoss(nn.Module):
    """Smooth reconstruction loss that remains stable near zero error."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((output - target) ** 2 + self.eps**2))


class EarlyStopping:
    """Stop after a minimized validation metric stops improving.

    Args:
        patience: Consecutive unimproved epochs to allow. Zero disables stopping.
        min_delta: Minimum metric decrease that counts as an improvement.

    Call the instance once per epoch. It returns ``True`` when training should
    stop and resets its counter whenever sufficient improvement is observed.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4) -> None:
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def __call__(self, metric: float) -> bool:
        """Update state with the latest metric and report whether to stop."""
        if metric < self.best - self.min_delta:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.patience > 0 and self.bad_epochs >= self.patience


def masked_charbonnier_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Measure restoration error primarily where synthetic damage was applied."""
    error = torch.sqrt((output - target) ** 2 + eps**2)
    weights = mask.expand_as(error)
    return torch.sum(error * weights) / weights.sum().clamp_min(1.0)


def _channelwise_kernel(kernel: torch.Tensor, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return kernel.to(device=device, dtype=dtype).view(1, 1, *kernel.shape).repeat(channels, 1, 1, 1)


def gradient_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    channels = output.shape[1]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    wx = _channelwise_kernel(kx, channels, output.device, output.dtype)
    wy = _channelwise_kernel(ky, channels, output.device, output.dtype)
    out_x = F.conv2d(output, wx, padding=1, groups=channels)
    out_y = F.conv2d(output, wy, padding=1, groups=channels)
    tgt_x = F.conv2d(target, wx, padding=1, groups=channels)
    tgt_y = F.conv2d(target, wy, padding=1, groups=channels)
    return F.l1_loss(out_x, tgt_x) + F.l1_loss(out_y, tgt_y)


def laplacian_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    channels = output.shape[1]
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    weight = _channelwise_kernel(kernel, channels, output.device, output.dtype)
    out_lap = F.conv2d(output, weight, padding=1, groups=channels)
    tgt_lap = F.conv2d(target, weight, padding=1, groups=channels)
    return F.l1_loss(out_lap, tgt_lap)


def frequency_l1_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    output_fft = torch.fft.rfft2(output, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    out_mag = torch.log1p(torch.abs(output_fft))
    tgt_mag = torch.log1p(torch.abs(target_fft))

    _, _, h, fw = out_mag.shape
    yy = torch.linspace(0.0, 1.0, h, device=output.device, dtype=output.dtype).view(1, 1, h, 1)
    xx = torch.linspace(0.0, 1.0, fw, device=output.device, dtype=output.dtype).view(1, 1, 1, fw)
    high_freq_weight = 0.35 + torch.sqrt(xx**2 + yy**2).clamp(0.0, 1.0)
    return torch.mean(torch.abs(out_mag - tgt_mag) * high_freq_weight)


def local_contrast_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out_mean = F.avg_pool2d(output, kernel_size=9, stride=1, padding=4)
    tgt_mean = F.avg_pool2d(target, kernel_size=9, stride=1, padding=4)
    out_detail = output - out_mean
    tgt_detail = target - tgt_mean
    return F.l1_loss(out_detail, tgt_detail)


TRAIN_CONFIG_KEYS = {
    "data",
    "epochs",
    "batch_size",
    "patch_size",
    "num_patches",
    "lr",
    "width",
    "middle_blocks",
    "workers",
    "color_mode",
    "screentone_probability",
    "artifact_weight",
    "freq_weight",
    "gradient_weight",
    "laplacian_weight",
    "contrast_weight",
    "checkpoint_dir",
    "sample_dir",
    "sample_every",
    "early_stop_patience",
    "early_stop_min_delta",
    "seed",
}


def get_config(path: str | Path) -> dict:
    """Load and validate training hyperparameters from a YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a YAML mapping: {config_path}")

    unknown = sorted(set(config) - TRAIN_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown training config keys: {', '.join(unknown)}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a manga screentone restoration model.")
    parser.add_argument("--config", default="config/default.yaml", help="YAML training configuration.")
    config_args, _ = parser.parse_known_args()
    parser.set_defaults(**get_config(config_args.config))

    parser.add_argument("--data", help="Clean image file or recursively scanned directory.")
    parser.add_argument("--epochs", type=int, help="Number of complete training epochs.")
    parser.add_argument("--batch-size", type=int, help="Patches processed per optimizer step.")
    parser.add_argument("--patch-size", type=int, help="Square training crop size in pixels.")
    parser.add_argument("--num-patches", type=int, help="Random synthetic patches generated per epoch.")
    parser.add_argument("--lr", type=float, help="Initial AdamW learning rate.")
    parser.add_argument("--width", type=int, help="Base model channel width.")
    parser.add_argument("--middle-blocks", type=int, help="Blocks in the model bottleneck.")
    parser.add_argument("--workers", type=int, help="DataLoader worker process count.")
    parser.add_argument("--color-mode", choices=("gray", "rgb"), help="Training image color mode.")
    parser.add_argument("--screentone-probability", type=float, help="Chance of adding synthetic clean screentone.")
    parser.add_argument("--artifact-weight", type=float, help="Extra loss weight inside corrupted regions.")
    parser.add_argument("--freq-weight", type=float, help="Frequency-detail loss weight.")
    parser.add_argument("--gradient-weight", type=float, help="Edge-gradient loss weight.")
    parser.add_argument("--laplacian-weight", type=float, help="Fine-detail Laplacian loss weight.")
    parser.add_argument("--contrast-weight", type=float, help="Local-contrast loss weight.")
    parser.add_argument("--checkpoint-dir", help="Checkpoint output directory.")
    parser.add_argument("--sample-dir", help="Preview image output directory.")
    parser.add_argument("--sample-every", type=int, help="Epoch interval between preview images.")
    parser.add_argument("--early-stop-patience", type=int, help="Unimproved epochs before stopping; zero disables it.")
    parser.add_argument("--early-stop-min-delta", type=float, help="Metric decrease required to reset patience.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible augmentation.")
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> DualDomainNAFNet:
    img_channel = 1 if args.color_mode == "gray" else 3
    return DualDomainNAFNet(
        img_channel=img_channel,
        width=args.width,
        middle_blk_num=args.middle_blocks,
    ).to(device)


def save_checkpoint(
    path: Path,
    model: DualDomainNAFNet,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "model_args": {
                "img_channel": model.img_channel,
                "width": args.width,
                "middle_blk_num": args.middle_blocks,
            },
            "train_args": vars(args),
        },
        path,
    )


def train() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training color mode: {args.color_mode}")

    checkpoint_dir = Path(args.checkpoint_dir)
    sample_dir = Path(args.sample_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    dataset = RestorationDataset(
        args.data,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        training=True,
        color_mode=args.color_mode,
        screentone_probability=args.screentone_probability,
        return_mask=True,
    )
    print(f"Discovered {len(dataset.paths)} training images recursively under: {args.data}")

    # Generate this batch once so preview images are directly comparable across epochs.
    preview_dataset = RestorationDataset(
        args.data,
        patch_size=args.patch_size,
        num_patches=min(4, len(dataset.paths)),
        training=False,
        color_mode=args.color_mode,
        screentone_probability=0.0,
        return_mask=True,
    )
    preview_input, preview_target, preview_mask = next(
        iter(DataLoader(preview_dataset, batch_size=len(preview_dataset), shuffle=False))
    )
    preview_input = preview_input.to(device)
    preview_target = preview_target.to(device)
    preview_mask = preview_mask.to(device)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model(args, device)
    criterion = CharbonnierLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(args.early_stop_patience, args.early_stop_min_delta)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, (input_img, target, artifact_mask) in enumerate(pbar):
            input_img = input_img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            artifact_mask = artifact_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(input_img)

            loss_charb = criterion(output, target)
            loss_artifact = masked_charbonnier_loss(output, target, artifact_mask)
            loss_grad = gradient_loss(output, target)
            loss_lap = laplacian_loss(output, target)
            loss_freq = frequency_l1_loss(output, target)
            loss_contrast = local_contrast_loss(output, target)
            loss = (
                loss_charb
                + args.artifact_weight * loss_artifact
                + args.gradient_weight * loss_grad
                + args.laplacian_weight * loss_lap
                + args.freq_weight * loss_freq
                + args.contrast_weight * loss_contrast
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.5f}",
                    "pix": f"{loss_charb.item():.5f}",
                    "artifact": f"{loss_artifact.item():.5f}",
                    "tone": f"{loss_contrast.item():.5f}",
                }
            )

        avg_loss = epoch_loss / max(1, len(dataloader))
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} avg loss: {avg_loss:.6f}; lr: {current_lr:.3e}")

        model.eval()
        with torch.no_grad():
            preview_output = model(preview_input).clamp(0.0, 1.0)
            preview_artifact_loss = masked_charbonnier_loss(
                preview_output, preview_target, preview_mask
            )
            if epoch == 1 or epoch % args.sample_every == 0:
                sample = torch.cat([preview_input, preview_output, preview_target], dim=0)
                save_image(sample, sample_dir / f"epoch_{epoch:04d}.png", nrow=len(preview_input))
        preview_loss = preview_artifact_loss.item()
        print(f"Fixed preview artifact loss: {preview_loss:.6f}")

        save_checkpoint(checkpoint_dir / "latest_model.pth", model, optimizer, epoch, avg_loss, args)
        if preview_loss < best_loss:
            best_loss = preview_loss
            save_checkpoint(checkpoint_dir / "best_model.pth", model, optimizer, epoch, preview_loss, args)
            print(f"Saved best model: {best_loss:.6f}")

        scheduler.step()
        if early_stopping(preview_loss):
            print(
                f"Early stopping at epoch {epoch}: preview loss did not improve "
                f"by {args.early_stop_min_delta:g} for {args.early_stop_patience} epochs."
            )
            break

    print("Training complete.")


if __name__ == "__main__":
    train()
