import random
import yaml
import torch

from pathlib import Path
from PIL import Image
from typing import Any, Callable, Optional, Union
from torchvision.transforms import functional as TF



IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

TRAIN_CONFIG_KEYS = {
    "data",
    "epochs",
    "batch_size",
    "patch_size",
    "num_patches",
    "validation_split",
    "lr",
    "width",
    "middle_blocks",
    "workers",
    "color_mode",
    "screentone",
    "artifact_augmentor",
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

DATASET_CONFIG_KEYS = {
    "screentone": {"probability", "region_count", "strength"},
    "artifact_augmentor": {
        "alpha_range",
        "stripe_count",
        "bar_count",
        "stain_count",
        "p_stripes",
        "p_bars",
        "p_stains",
    },
}


def list_image_paths(path: Union[str, Path]) -> list[Path]:
    """Return supported images from a file or recursively from a directory."""
    root = Path(path)
    if root.is_file():
        if root.suffix.lower() not in IMG_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {root}")
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Image path does not exist: {root}")
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No images found under: {root}")
    return paths


def pil_to_tensor(image: Image.Image, color_mode: str = "gray") -> torch.Tensor:
    if color_mode == "gray":
        return TF.to_tensor(image.convert("L"))
    if color_mode == "rgb":
        return TF.to_tensor(image.convert("RGB"))
    raise ValueError(f"Unsupported color_mode: {color_mode}")

def split_image_paths(
    paths: list[Path], validation_split: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Deterministically split clean source pages into training and validation sets."""
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in the range [0, 1)")
    if len(paths) < 2 or validation_split == 0.0:
        print("Validation split disabled or impossible; validation reuses training pages.")
        return paths, paths

    shuffled = paths.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = min(len(paths) - 1, max(1, round(len(paths) * validation_split)))
    return shuffled[validation_count:], shuffled[:validation_count]


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
    for section, allowed_keys in DATASET_CONFIG_KEYS.items():
        values = config.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"Training config section '{section}' must be a mapping")
        unknown = sorted(set(values) - allowed_keys)
        if unknown:
            raise ValueError(
                f"Unknown keys in training config section '{section}': "
                f"{', '.join(unknown)}"
            )
    return config