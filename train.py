import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import RestorationDataset
from model import DualDomainNAFNet


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((output - target) ** 2 + self.eps**2))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a manga screentone restoration model.")
    parser.add_argument("--data", default="data", help="Clean manga image file or directory used to synthesize training pairs.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-patches", type=int, default=1000, help="Virtual samples per epoch.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--middle-blocks", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--color-mode", choices=("gray", "rgb"), default="gray")
    parser.add_argument("--screentone-probability", type=float, default=0.35)
    parser.add_argument("--freq-weight", type=float, default=0.03)
    parser.add_argument("--gradient-weight", type=float, default=0.08)
    parser.add_argument("--laplacian-weight", type=float, default=0.04)
    parser.add_argument("--contrast-weight", type=float, default=0.05)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--sample-dir", default="samples")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
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
    )
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

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, (input_img, target) in enumerate(pbar):
            input_img = input_img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(input_img).clamp(0.0, 1.0)

            loss_charb = criterion(output, target)
            loss_grad = gradient_loss(output, target)
            loss_lap = laplacian_loss(output, target)
            loss_freq = frequency_l1_loss(output, target)
            loss_contrast = local_contrast_loss(output, target)
            loss = (
                loss_charb
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
                    "tone": f"{loss_contrast.item():.5f}",
                }
            )

            if step == 0 and (epoch == 1 or epoch % args.sample_every == 0):
                with torch.no_grad():
                    sample = torch.cat([input_img[:4], output[:4], target[:4]], dim=0)
                    save_image(sample, sample_dir / f"epoch_{epoch:04d}.png", nrow=min(4, input_img.size(0)))

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch} avg loss: {avg_loss:.6f}")

        save_checkpoint(checkpoint_dir / "latest_model.pth", model, optimizer, epoch, avg_loss, args)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(checkpoint_dir / "best_model.pth", model, optimizer, epoch, avg_loss, args)
            print(f"Saved best model: {best_loss:.6f}")

    print("Training complete.")


if __name__ == "__main__":
    train()
