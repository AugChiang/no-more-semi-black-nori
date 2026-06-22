import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import BlackArtifactAugmentor
from model import DualDomainNAFNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore monochrome manga with semi-transparent black artifacts.")
    parser.add_argument("--input", default="data/input_0001.png", help="Corrupted image to restore.")
    parser.add_argument("--output", default="restored_output.png")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--mode", choices=("auto", "model", "classical"), default="auto")
    parser.add_argument("--color-mode", choices=("auto", "gray", "rgb"), default="auto")
    parser.add_argument("--width", type=int, default=32, help="Model width if checkpoint has no metadata.")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for model inference. Use 0 for full image.")
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--add-synthetic", action="store_true", help="Add synthetic artifacts before restoration.")
    parser.add_argument("--synthetic-output", default="stained_input.png")
    return parser.parse_args()


def _checkpoint_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Unsupported checkpoint format")


def _checkpoint_img_channel(checkpoint: object, fallback: int = 1) -> int:
    if isinstance(checkpoint, dict) and "model_args" in checkpoint:
        model_args = checkpoint.get("model_args", {})
        if "img_channel" in model_args:
            return int(model_args["img_channel"])
    state = _checkpoint_state(checkpoint)
    intro = state.get("intro.weight")
    if intro is not None and intro.ndim == 4:
        return int(intro.shape[1])
    return fallback


def load_checkpoint(path: Path, device: torch.device, fallback_width: int, fallback_channel: int) -> DualDomainNAFNet:
    checkpoint = torch.load(path, map_location=device)
    img_channel = _checkpoint_img_channel(checkpoint, fallback=fallback_channel)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_args = checkpoint.get("model_args", {})
        model = DualDomainNAFNet(
            img_channel=img_channel,
            width=int(model_args.get("width", fallback_width)),
            middle_blk_num=int(model_args.get("middle_blk_num", 1)),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
    else:
        model = DualDomainNAFNet(img_channel=img_channel, width=fallback_width).to(device)
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def image_to_tensor(image: Image.Image, device: torch.device, channels: int) -> torch.Tensor:
    if channels == 1:
        arr = np.array(image.convert("L"))
        tensor = torch.from_numpy(arr).unsqueeze(0).float() / 255.0
    elif channels == 3:
        arr = np.array(image.convert("RGB"))
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    else:
        raise ValueError(f"Unsupported channel count: {channels}")
    return tensor.unsqueeze(0).to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    if tensor.shape[0] == 1:
        arr = tensor.squeeze(0).numpy() * 255.0
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    arr = tensor.permute(1, 2, 0).numpy() * 255.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def infer_full(model: DualDomainNAFNet, tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(tensor).clamp(0.0, 1.0)


def infer_tiled(model: DualDomainNAFNet, tensor: torch.Tensor, tile_size: int, overlap: int) -> torch.Tensor:
    if tile_size <= 0:
        return infer_full(model, tensor)

    _, _, h, w = tensor.shape
    if h <= tile_size and w <= tile_size:
        return infer_full(model, tensor)

    stride = max(1, tile_size - overlap)
    output = torch.zeros_like(tensor)
    weight = torch.zeros_like(tensor)

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        values = list(range(0, length - tile_size + 1, stride))
        last = length - tile_size
        if values[-1] != last:
            values.append(last)
        return values

    ys = starts(h)
    xs = starts(w)

    window = torch.ones((1, 1, tile_size, tile_size), device=tensor.device, dtype=tensor.dtype)
    if overlap > 0:
        ramp = torch.linspace(0.1, 1.0, steps=min(overlap, tile_size // 2), device=tensor.device, dtype=tensor.dtype)
        blend = torch.ones(tile_size, device=tensor.device, dtype=tensor.dtype)
        blend[: ramp.numel()] = ramp
        blend[-ramp.numel() :] = ramp.flip(0)
        window = blend.view(1, 1, tile_size, 1) * blend.view(1, 1, 1, tile_size)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = tensor[:, :, y : y + tile_size, x : x + tile_size]
                pad_h = tile_size - patch.shape[-2]
                pad_w = tile_size - patch.shape[-1]
                if pad_h or pad_w:
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="replicate")
                pred = model(patch).clamp(0.0, 1.0)[:, :, : tile_size - pad_h, : tile_size - pad_w]
                ww = window[:, :, : pred.shape[-2], : pred.shape[-1]]
                output[:, :, y : y + pred.shape[-2], x : x + pred.shape[-1]] += pred * ww
                weight[:, :, y : y + pred.shape[-2], x : x + pred.shape[-1]] += ww

    return output / weight.clamp_min(1e-6)


def classical_restore(image: Image.Image) -> Image.Image:
    gray = np.array(image.convert("L")).astype(np.float32) / 255.0
    h, w = gray.shape
    k = max(31, (min(h, w) // 10) | 1)

    local_ref = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    local_ref = cv2.GaussianBlur(local_ref, (k, k), 0)
    attenuation = (local_ref - gray) / np.maximum(local_ref, 0.08)
    attenuation = np.clip(attenuation - 0.03, 0.0, 0.78)
    attenuation = cv2.GaussianBlur(attenuation, (max(15, (k // 3) | 1), max(15, (k // 3) | 1)), 0)

    restored = gray / np.maximum(1.0 - attenuation, 0.22)
    restored = np.clip(restored, 0.0, 1.0)
    return Image.fromarray((restored * 255.0).astype(np.uint8), mode="L")


def maybe_add_synthetic(image: Image.Image, output_path: Optional[Path], channels: int) -> Image.Image:
    if channels == 1:
        arr = np.array(image.convert("L"))
    else:
        arr = np.array(image.convert("RGB"))
    arr = BlackArtifactAugmentor().add_to_numpy(arr)
    stained = Image.fromarray(arr)
    if output_path is not None:
        stained.save(output_path)
        print(f"Saved synthetic corrupted input to {output_path}")
    return stained


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    checkpoint_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fallback_channel = 1 if args.color_mode in ("auto", "gray") else 3
    use_model = args.mode == "model" or (args.mode == "auto" and checkpoint_path.exists())
    image = Image.open(input_path)

    if use_model:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = load_checkpoint(checkpoint_path, device, args.width, fallback_channel)
        channels = model.img_channel
        if args.add_synthetic:
            image = maybe_add_synthetic(image, Path(args.synthetic_output), channels)
        tensor = image_to_tensor(image, device, channels)
        restored = tensor_to_image(infer_tiled(model, tensor, args.tile_size, args.overlap))
        print(f"Restored with model checkpoint: {checkpoint_path}")
    else:
        if args.add_synthetic:
            image = maybe_add_synthetic(image, Path(args.synthetic_output), fallback_channel)
        restored = classical_restore(image)
        print("Restored with grayscale classical fallback. Train a checkpoint for learned restoration.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    restored.save(output_path)
    print(f"Saved restored image to {output_path}")


if __name__ == "__main__":
    main()
