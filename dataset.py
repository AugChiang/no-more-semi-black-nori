from pathlib import Path
import random
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_image_paths(path: Union[str, Path]) -> list[Path]:
    root = Path(path)
    if root.is_file():
        if root.suffix.lower() not in IMG_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {root}")
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Image path does not exist: {root}")
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)
    if not paths:
        raise ValueError(f"No images found under: {root}")
    return paths


def pil_to_tensor(image: Image.Image, color_mode: str = "gray") -> torch.Tensor:
    if color_mode == "gray":
        return TF.to_tensor(image.convert("L"))
    if color_mode == "rgb":
        return TF.to_tensor(image.convert("RGB"))
    raise ValueError(f"Unsupported color_mode: {color_mode}")


class ScreentoneSynthesizer:
    """Optionally injects clean dot/line tones into light regions of training targets."""

    def __init__(
        self,
        probability: float = 0.35,
        region_count: Tuple[int, int] = (1, 4),
        strength: Tuple[float, float] = (0.08, 0.32),
    ) -> None:
        self.probability = probability
        self.region_count = region_count
        self.strength = strength

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.probability <= 0.0 or random.random() > self.probability:
            return image

        channels, h, w = image.shape
        base = image.mean(dim=0).cpu().numpy()
        writable = np.clip((base - 0.30) / 0.55, 0.0, 1.0).astype(np.float32)
        if writable.max() <= 1e-3:
            return image

        yy, xx = np.mgrid[:h, :w]
        tone = np.zeros((h, w), dtype=np.float32)

        for _ in range(random.randint(*self.region_count)):
            region = np.zeros((h, w), dtype=np.float32)
            if random.random() < 0.65:
                center = (random.randint(0, w - 1), random.randint(0, h - 1))
                axes = (
                    random.randint(max(8, w // 12), max(10, w // 3)),
                    random.randint(max(8, h // 12), max(10, h // 3)),
                )
                cv2.ellipse(region, center, axes, random.uniform(0, 180), 0, 360, 1.0, -1)
            else:
                x0 = random.randint(0, max(0, w - 16))
                y0 = random.randint(0, max(0, h - 16))
                x1 = random.randint(x0 + 8, w)
                y1 = random.randint(y0 + 8, h)
                cv2.rectangle(region, (x0, y0), (x1, y1), 1.0, -1)

            period = random.randint(4, 12)
            angle = np.deg2rad(random.choice((0, 15, 30, 45, 60, 75, 90)))
            xr = xx * np.cos(angle) + yy * np.sin(angle)
            yr = -xx * np.sin(angle) + yy * np.cos(angle)

            if random.random() < 0.65:
                radius = random.uniform(0.16, 0.34) * period
                dots = ((xr % period - period / 2) ** 2 + (yr % period - period / 2) ** 2) < radius**2
                pattern = dots.astype(np.float32)
            else:
                duty = random.uniform(0.18, 0.36) * period
                pattern = ((xr % period) < duty).astype(np.float32)

            edge = cv2.GaussianBlur(region, (9, 9), 0)
            alpha = random.uniform(*self.strength)
            tone = np.maximum(tone, edge * pattern * alpha)

        tone *= writable
        tone_t = torch.from_numpy(tone).to(device=image.device, dtype=image.dtype).unsqueeze(0)
        if channels > 1:
            tone_t = tone_t.expand(channels, -1, -1)
        return (image * (1.0 - tone_t)).clamp(0.0, 1.0)


class BlackArtifactAugmentor:
    """Adds translucent black overlays resembling stripes, bars, and stains."""

    def __init__(
        self,
        alpha_range: Tuple[float, float] = (0.10, 0.72),
        stripe_count: Tuple[int, int] = (1, 8),
        bar_count: Tuple[int, int] = (0, 4),
        stain_count: Tuple[int, int] = (0, 4),
        p_stripes: float = 0.88,
        p_bars: float = 0.58,
        p_stains: float = 0.45,
    ) -> None:
        self.alpha_range = alpha_range
        self.stripe_count = stripe_count
        self.bar_count = bar_count
        self.stain_count = stain_count
        self.p_stripes = p_stripes
        self.p_bars = p_bars
        self.p_stains = p_stains

    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 3:
            raise ValueError("Expected a CHW tensor")

        _, h, w = image.shape
        mask = np.zeros((h, w), dtype=np.float32)

        if random.random() < self.p_stripes:
            self._add_stripes(mask)
        if random.random() < self.p_bars:
            self._add_bars(mask)
        if random.random() < self.p_stains:
            self._add_stains(mask)
        if mask.max() <= 1e-6:
            self._add_stripes(mask)

        mask = np.clip(mask, 0.0, 0.92)
        mask_t = torch.from_numpy(mask).to(device=image.device, dtype=image.dtype).unsqueeze(0)
        corrupted = image * (1.0 - mask_t)
        return corrupted.clamp(0.0, 1.0), mask_t

    def add_to_numpy(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            corrupted, _ = self(tensor)
            out = corrupted.squeeze(0).numpy() * 255.0
        elif image.ndim == 3:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            corrupted, _ = self(tensor)
            out = corrupted.permute(1, 2, 0).numpy() * 255.0
        else:
            raise ValueError("Expected HxW or HxWxC numpy image")
        return np.clip(out, 0, 255).astype(np.uint8)

    def _alpha(self, scale: float = 1.0) -> float:
        lo, hi = self.alpha_range
        return random.uniform(lo, hi) * scale

    def _soften(self, layer: np.ndarray, k: int) -> np.ndarray:
        k = max(3, int(k) | 1)
        return cv2.GaussianBlur(layer, (k, k), 0)

    def _add_stripes(self, mask: np.ndarray) -> None:
        h, w = mask.shape
        diag = int(np.ceil(np.hypot(h, w)))
        canvas_size = diag * 2
        count = random.randint(*self.stripe_count)
        shared_angle = random.choice((0, 90, random.uniform(-80.0, 80.0)))

        for _ in range(count):
            width = random.randint(max(3, min(h, w) // 180), max(8, min(h, w) // 16))
            layer = np.zeros((canvas_size, canvas_size), dtype=np.float32)
            cx = random.randint(canvas_size // 5, canvas_size * 4 // 5)
            cv2.rectangle(layer, (cx, 0), (cx + width, canvas_size), self._alpha(), -1)
            angle = shared_angle + random.uniform(-4.0, 4.0)
            matrix = cv2.getRotationMatrix2D((canvas_size / 2, canvas_size / 2), angle, 1.0)
            layer = cv2.warpAffine(layer, matrix, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR)
            y0 = canvas_size // 2 - h // 2
            x0 = canvas_size // 2 - w // 2
            layer = layer[y0 : y0 + h, x0 : x0 + w]
            layer = self._soften(layer, max(3, width // 2))
            np.maximum(mask, layer, out=mask)

    def _add_bars(self, mask: np.ndarray) -> None:
        h, w = mask.shape
        for _ in range(random.randint(*self.bar_count)):
            layer = np.zeros_like(mask)
            horizontal = random.random() < 0.5
            if horizontal:
                bar_h = random.randint(max(8, h // 35), max(12, h // 5))
                y = random.randint(-bar_h // 2, h - 1)
                cv2.rectangle(layer, (0, y), (w, y + bar_h), self._alpha(0.8), -1)
                blur = max(5, bar_h // 5)
            else:
                bar_w = random.randint(max(8, w // 45), max(12, w // 6))
                x = random.randint(-bar_w // 2, w - 1)
                cv2.rectangle(layer, (x, 0), (x + bar_w, h), self._alpha(0.8), -1)
                blur = max(5, bar_w // 5)
            layer = self._soften(layer, blur)
            np.maximum(mask, layer, out=mask)

    def _add_stains(self, mask: np.ndarray) -> None:
        h, w = mask.shape
        for _ in range(random.randint(*self.stain_count)):
            layer = np.zeros_like(mask)
            rx = random.randint(max(8, w // 35), max(10, w // 5))
            ry = random.randint(max(8, h // 35), max(10, h // 5))
            center = (random.randint(0, w - 1), random.randint(0, h - 1))
            cv2.ellipse(layer, center, (rx, ry), random.uniform(0, 180), 0, 360, self._alpha(0.65), -1)
            noise = cv2.resize(
                np.random.rand(max(4, h // 28), max(4, w // 28)).astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_CUBIC,
            )
            layer *= np.clip(0.50 + noise, 0.0, 1.0)
            layer = self._soften(layer, max(9, min(rx, ry) // 2))
            np.maximum(mask, layer, out=mask)


class RestorationDataset(Dataset):
    """Clean manga dataset returning synthetic-corrupted input/target pairs."""

    def __init__(
        self,
        image_path: Union[str, Path],
        patch_size: int = 256,
        num_patches: Optional[int] = None,
        training: bool = True,
        color_mode: str = "gray",
        screentone_probability: float = 0.35,
        augmentor: Optional[BlackArtifactAugmentor] = None,
    ) -> None:
        self.paths = list_image_paths(image_path)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.training = training
        self.color_mode = color_mode
        self.augmentor = augmentor or BlackArtifactAugmentor()
        self.screentone_synth = ScreentoneSynthesizer(probability=screentone_probability)

    def __len__(self) -> int:
        if self.num_patches is not None:
            return self.num_patches
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[index % len(self.paths)]
        image = Image.open(path)
        image = self._prepare_patch(image)
        target = pil_to_tensor(image, self.color_mode)
        if self.training:
            target = self.screentone_synth(target)
        corrupted, _ = self.augmentor(target)
        return corrupted, target

    def _prepare_patch(self, image: Image.Image) -> Image.Image:
        if self.training:
            image = self._random_crop_or_resize(image, self.patch_size)
            if random.random() < 0.5:
                image = TF.hflip(image)
            if random.random() < 0.5:
                image = TF.vflip(image)
        else:
            image = self._center_crop_or_resize(image, self.patch_size)
        return image

    def _random_crop_or_resize(self, image: Image.Image, size: int) -> Image.Image:
        w, h = image.size
        scale = max(size / min(w, h), 1.0)
        if scale > 1.0:
            image = image.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)
            w, h = image.size
        left = random.randint(0, w - size)
        top = random.randint(0, h - size)
        return image.crop((left, top, left + size, top + size))

    def _center_crop_or_resize(self, image: Image.Image, size: int) -> Image.Image:
        w, h = image.size
        scale = max(size / min(w, h), 1.0)
        if scale > 1.0:
            image = image.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)
            w, h = image.size
        left = max(0, (w - size) // 2)
        top = max(0, (h - size) // 2)
        return image.crop((left, top, left + size, top + size))


StripeAugmentor = BlackArtifactAugmentor
MangaStripeDataset = RestorationDataset
