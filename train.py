import argparse
from contextlib import nullcontext
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
from nudenet_regions import load_nudenet_regions
from pretrained import load_nafnet_pretrained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a manga screentone restoration model.")
    parser.add_argument("--config", default="config/default.yaml", help="YAML training configuration.")
    config_args, _ = parser.parse_known_args()
    defaults = {
        "nafnet_pretrained_weights": None,
        "nudenet_weights": None,
        "nudenet_cache": "data/nudenet_regions.json",
        "nudenet_confidence": 0.15,
        "nudenet_resolution": 640,
        "nudenet_batch_size": 8,
        "region_crop_probability": 1.0,
    }
    defaults.update(get_config(config_args.config))
    parser.set_defaults(**defaults)

    parser.add_argument("--data", help="Clean image file or recursively scanned directory.")
    parser.add_argument("--epochs", type=int, help="Number of complete training epochs.")
    parser.add_argument("--batch-size", type=int, help="Patches processed per optimizer step.")
    parser.add_argument("--patch-size", type=int, help="Square training crop size in pixels.")
    parser.add_argument("--num-patches", type=int, help="Random synthetic patches generated per epoch.")
    parser.add_argument("--validation-split", type=float, help="Fraction of clean pages reserved for validation.")
    parser.add_argument("--lr", type=float, help="Initial AdamW learning rate.")
    parser.add_argument("--width", type=int, help="Base model channel width.")
    parser.add_argument("--middle-blocks", type=int, help="Blocks in the model bottleneck.")
    parser.add_argument("--encoder-blocks", type=int, nargs="+", help="NAF blocks in each encoder stage.")
    parser.add_argument("--decoder-blocks", type=int, nargs="+", help="NAF blocks in each decoder stage.")
    parser.add_argument("--use-fft", action=argparse.BooleanOptionalAction,
                        help="Enable the legacy FFT bottleneck.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction,
                        help="Use CUDA mixed precision when available.")
    parser.add_argument("--workers", type=int, help="DataLoader worker process count.")
    parser.add_argument("--color-mode", choices=("gray", "rgb"), help="Training image color mode.")
    parser.add_argument("--nafnet-pretrained-weights", help="Official NAFNet weights used to initialize new training runs.")
    parser.add_argument("--no-pretrained", action="store_true", help="Start with random restoration-model weights.")
    parser.add_argument("--nudenet-weights", help="NudeNet 640m .pt weights for guided masks.")
    parser.add_argument("--nudenet-cache", help="JSON cache for clean-image region detections.")
    parser.add_argument("--nudenet-confidence", type=float, help="Minimum genital detection confidence.")
    parser.add_argument("--nudenet-resolution", type=int, help="NudeNet inference image size.")
    parser.add_argument("--nudenet-batch-size", type=int, help="Images per NudeNet detection batch.")
    parser.add_argument("--region-crop-probability", type=float, help="Chance of centering a crop on a detected region.")
    parser.add_argument("--artifact-weight", type=float, help="Extra loss weight inside corrupted regions.")
    parser.add_argument("--freq-weight", type=float, help="Frequency-detail loss weight.")
    parser.add_argument("--gradient-weight", type=float, help="Edge-gradient loss weight.")
    parser.add_argument("--laplacian-weight", type=float, help="Fine-detail Laplacian loss weight.")
    parser.add_argument("--contrast-weight", type=float, help="Local-contrast loss weight.")
    parser.add_argument("--checkpoint-dir", help="Checkpoint output directory.")
    parser.add_argument("--sample-dir", help="Preview image output directory.")
    parser.add_argument("--sample-every", type=int, help="Epoch interval between preview images.")
    parser.add_argument("--resume", help="Checkpoint path to resume training from.")
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
        enc_blk_nums=tuple(args.encoder_blocks),
        dec_blk_nums=tuple(args.decoder_blocks),
        use_fft=args.use_fft,
    ).to(device)


def save_checkpoint(
    path: Path,
    model: DualDomainNAFNet,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    loss: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "model_args": {
                "img_channel": model.img_channel,
                "width": args.width,
                "middle_blk_num": args.middle_blocks,
                "enc_blk_nums": list(args.encoder_blocks),
                "dec_blk_nums": list(args.decoder_blocks),
                "use_fft": args.use_fft,
            },
            "train_args": vars(args),
        },
        path,
    )


def load_history(path: Path) -> list[float]:
    if not path.is_file():
        return []
    return np.load(path).astype(np.float32).tolist()


def restore_training_state(
    resume_path: Path,
    model: DualDomainNAFNet,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(resume_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unsupported training checkpoint format: {resume_path}")

    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        for _ in range(int(checkpoint.get("epoch", 0))):
            scheduler.step()
    if "scaler" in checkpoint and scaler.is_enabled():
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_loss = float(checkpoint.get("loss", float("inf")))
    return start_epoch, best_loss


def apply_resume_model_args(args: argparse.Namespace) -> None:
    """Use the saved architecture when resuming an older or customized model."""
    checkpoint = torch.load(args.resume, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unsupported training checkpoint format: {args.resume}")
    model_args = checkpoint.get("model_args", {})
    state = checkpoint["model"]
    args.width = int(model_args.get("width", args.width))
    args.middle_blocks = int(model_args.get("middle_blk_num", args.middle_blocks))
    args.encoder_blocks = list(model_args.get("enc_blk_nums", (1, 1, 1, 1)))
    args.decoder_blocks = list(model_args.get("dec_blk_nums", (1, 1, 1, 1)))
    args.use_fft = bool(model_args.get(
        "use_fft", any(key.startswith("fft_block.") for key in state)
    ))
    channels = int(model_args.get("img_channel", state["intro.weight"].shape[1]))
    if channels not in (1, 3):
        raise ValueError(f"Unsupported checkpoint channel count: {channels}")
    args.color_mode = "gray" if channels == 1 else "rgb"


def train() -> None:
    args = parse_args()
    if args.resume:
        apply_resume_model_args(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )

    def autocast_context():
        return torch.autocast("cuda", dtype=amp_dtype) if amp_enabled else nullcontext()

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"AMP: {str(amp_dtype).split('.')[-1] if amp_enabled else 'off'}")
    print(f"Training color mode: {args.color_mode}")

    checkpoint_dir = Path(args.checkpoint_dir)
    sample_dir = Path(args.sample_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    all_paths = list_image_paths(args.data)
    if not args.nudenet_weights:
        raise ValueError("Guided-only masks require nudenet_weights in the config")
    regions_by_path = load_nudenet_regions(
        all_paths,
        weights=Path(args.nudenet_weights),
        cache_path=Path(args.nudenet_cache),
        confidence=args.nudenet_confidence,
        resolution=args.nudenet_resolution,
        batch_size=args.nudenet_batch_size,
    )
    guided_paths = [
        path for path in all_paths if regions_by_path.get(str(path.resolve()))
    ]
    if not guided_paths:
        raise ValueError("NudeNet found no target regions; guided-only training has no masks")
    train_paths, validation_paths = split_image_paths(
        guided_paths, args.validation_split, args.seed
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
        regions_by_path=regions_by_path,
        region_crop_probability=args.region_crop_probability,
    )
    print(
        f"Discovered {len(all_paths)} images under {args.data}; "
        f"{len(guided_paths)} with NudeNet regions: "
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
        regions_by_path=regions_by_path,
        region_crop_probability=args.region_crop_probability,
        validation_seed=args.seed,
    )
    validation_batches = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
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
    if not args.resume and not args.no_pretrained and args.nafnet_pretrained_weights:
        pretrained_path = Path(args.nafnet_pretrained_weights)
        load_nafnet_pretrained(model, pretrained_path)
        print(f"Initialized NAFNet from {pretrained_path}.")
    criterion = CharbonnierLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(args.early_stop_patience, args.early_stop_min_delta)

    start_epoch = 1
    best_loss = float("inf")
    training_history = load_history(checkpoint_dir / "training_loss.npy") if args.resume else []
    validation_history = load_history(checkpoint_dir / "validation_loss.npy") if args.resume else []

    if args.resume:
        resume_path = Path(args.resume)
        start_epoch, checkpoint_loss = restore_training_state(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        scheduler.T_max = args.epochs
        if not training_history:
            training_history = load_history(resume_path.parent / "training_loss.npy")
        if not validation_history:
            validation_history = load_history(resume_path.parent / "validation_loss.npy")
        if not training_history:
            training_history = load_history(resume_path.parent.parent / "training_loss.npy")
        if not validation_history:
            validation_history = load_history(resume_path.parent.parent / "validation_loss.npy")
        if validation_history:
            best_loss = min(validation_history)
        else:
            best_loss = checkpoint_loss
        print(f"Resumed from {resume_path} at epoch {start_epoch}.")

    if start_epoch > args.epochs:
        print(
            f"Checkpoint already reached epoch {start_epoch - 1}; "
            f"target epochs is {args.epochs}."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, (input_img, target, artifact_mask) in enumerate(pbar):
            input_img = input_img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            artifact_mask = artifact_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                output = model(input_img)
            loss, components = restoration_loss(
                output.float(), target.float(), artifact_mask.float(), criterion, args
            )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

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
        preview_sample = None
        save_preview = epoch == 1 or epoch % args.sample_every == 0
        with torch.inference_mode():
            for validation_input, validation_target, validation_mask in validation_batches:
                validation_input = validation_input.to(device, non_blocking=True)
                validation_target = validation_target.to(device, non_blocking=True)
                validation_mask = validation_mask.to(device, non_blocking=True)
                with autocast_context():
                    validation_output = model(validation_input)
                batch_loss, _ = restoration_loss(
                    validation_output.float(),
                    validation_target.float(),
                    validation_mask.float(),
                    criterion,
                    args,
                )
                batch_size = validation_input.size(0)
                validation_sum += batch_loss.item() * batch_size
                validation_samples += batch_size
                if save_preview and preview_sample is None:
                    preview_count = min(4, batch_size)
                    preview_sample = torch.cat([
                        validation_input[:preview_count],
                        validation_output[:preview_count].float().clamp(0.0, 1.0),
                        validation_target[:preview_count],
                    ], dim=0).cpu()

            validation_loss = validation_sum / validation_samples
            if preview_sample is not None:
                save_image(
                    preview_sample,
                    sample_dir / f"epoch_{epoch:04d}.png",
                    nrow=preview_count,
                )
        training_history.append(avg_loss)
        validation_history.append(validation_loss)
        np.save(checkpoint_dir / "training_loss.npy", np.asarray(training_history, dtype=np.float32))
        np.save(checkpoint_dir / "validation_loss.npy", np.asarray(validation_history, dtype=np.float32))
        print(f"Validation loss: {validation_loss:.6f}")

        scheduler.step()
        save_checkpoint(
            checkpoint_dir / "latest_model.pth", model, optimizer, scheduler,
            scaler, epoch, avg_loss, args,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(
                checkpoint_dir / "best_model.pth", model, optimizer, scheduler,
                scaler, epoch, validation_loss, args,
            )
            print(f"Saved best model: {best_loss:.6f}")

        if early_stopping(validation_loss):
            print(
                f"Early stopping at epoch {epoch}: preview loss did not improve "
                f"by {args.early_stop_min_delta:g} for {args.early_stop_patience} epochs."
            )
            break

    print("Training complete.")


if __name__ == "__main__":
    train()
