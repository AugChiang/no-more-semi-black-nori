import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import (
    BlackArtifactAugmentor,
    RestorationDataset,
    ScreentoneSynthesizer,
    list_image_paths,
)
from losses import CharbonnierLoss, restoration_loss
from utils import get_config, split_image_paths
from early_stopper import EarlyStopping
from model import DualDomainNAFNet


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
    parser.add_argument("--validation-split", type=float, help="Fraction of clean pages reserved for validation.")
    parser.add_argument("--lr", type=float, help="Initial AdamW learning rate.")
    parser.add_argument("--width", type=int, help="Base model channel width.")
    parser.add_argument("--middle-blocks", type=int, help="Blocks in the model bottleneck.")
    parser.add_argument("--workers", type=int, help="DataLoader worker process count.")
    parser.add_argument("--color-mode", choices=("gray", "rgb"), help="Training image color mode.")
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

    all_paths = list_image_paths(args.data)
    train_paths, validation_paths = split_image_paths(
        all_paths, args.validation_split, args.seed
    )
    screentone_synthesizer = ScreentoneSynthesizer(**args.screentone)
    artifact_augmentor = BlackArtifactAugmentor(**args.artifact_augmentor)
    dataset = RestorationDataset(
        args.data,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        training=True,
        color_mode=args.color_mode,
        augmentor=artifact_augmentor,
        screentone_synthesizer=screentone_synthesizer,
        return_mask=True,
        image_paths=train_paths,
    )
    print(
        f"Discovered {len(all_paths)} images under {args.data}: "
        f"{len(train_paths)} training, {len(validation_paths)} validation"
    )

    validation_dataset = RestorationDataset(
        args.data,
        patch_size=args.patch_size,
        num_patches=None,
        training=False,
        color_mode=args.color_mode,
        augmentor=artifact_augmentor,
        screentone_synthesizer=screentone_synthesizer,
        return_mask=True,
        image_paths=validation_paths,
    )
    # Cache corruptions once so validation metrics are directly comparable by epoch.
    validation_batches = list(
        DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
    )
    preview_input, preview_target, _ = validation_batches[0]
    preview_input = preview_input[:4].to(device)
    preview_target = preview_target[:4].to(device)
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
    training_history: list[float] = []
    validation_history: list[float] = []

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

            loss, components = restoration_loss(
                output, target, artifact_mask, criterion, args
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.5f}",
                    "pix": f"{components['pixel'].item():.5f}",
                    "artifact": f"{components['artifact'].item():.5f}",
                    "tone": f"{components['contrast'].item():.5f}",
                }
            )

        avg_loss = epoch_loss / max(1, len(dataloader))
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} avg loss: {avg_loss:.6f}; lr: {current_lr:.3e}")

        model.eval()
        validation_sum = 0.0
        validation_samples = 0
        with torch.no_grad():
            for validation_input, validation_target, validation_mask in validation_batches:
                validation_input = validation_input.to(device, non_blocking=True)
                validation_target = validation_target.to(device, non_blocking=True)
                validation_mask = validation_mask.to(device, non_blocking=True)
                validation_output = model(validation_input)
                batch_loss, _ = restoration_loss(
                    validation_output,
                    validation_target,
                    validation_mask,
                    criterion,
                    args,
                )
                batch_size = validation_input.size(0)
                validation_sum += batch_loss.item() * batch_size
                validation_samples += batch_size

            validation_loss = validation_sum / validation_samples
            if epoch == 1 or epoch % args.sample_every == 0:
                preview_output = model(preview_input).clamp(0.0, 1.0)
                sample = torch.cat([preview_input, preview_output, preview_target], dim=0)
                save_image(sample, sample_dir / f"epoch_{epoch:04d}.png", nrow=len(preview_input))
        training_history.append(avg_loss)
        validation_history.append(validation_loss)
        np.save(checkpoint_dir / "training_loss.npy", np.asarray(training_history, dtype=np.float32))
        np.save(checkpoint_dir / "validation_loss.npy", np.asarray(validation_history, dtype=np.float32))
        print(f"Validation loss: {validation_loss:.6f}")

        save_checkpoint(checkpoint_dir / "latest_model.pth", model, optimizer, epoch, avg_loss, args)
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(checkpoint_dir / "best_model.pth", model, optimizer, epoch, validation_loss, args)
            print(f"Saved best model: {best_loss:.6f}")

        scheduler.step()
        if early_stopping(validation_loss):
            print(
                f"Early stopping at epoch {epoch}: preview loss did not improve "
                f"by {args.early_stop_min_delta:g} for {args.early_stop_patience} epochs."
            )
            break

    print("Training complete.")


if __name__ == "__main__":
    train()
