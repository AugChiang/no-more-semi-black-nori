"""Cache NudeNet genital-region detections for clean training images."""

import json
from pathlib import Path
from typing import Sequence


TARGET_LABELS = {"FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"}
RegionBox = tuple[float, float, float, float]


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_nudenet_regions(
    image_paths: Sequence[Path],
    weights: Path,
    cache_path: Path,
    confidence: float = 0.15,
    resolution: int = 640,
    batch_size: int = 8,
) -> dict[str, list[RegionBox]]:
    """Detect once per clean image and reuse boxes until images or settings change."""
    weights = weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"NudeNet weights not found: {weights}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if resolution <= 0 or batch_size <= 0:
        raise ValueError("resolution and batch_size must be positive")

    settings = {
        "version": 1,
        "weights": str(weights),
        "weights_signature": _file_signature(weights),
        "confidence": confidence,
        "resolution": resolution,
        "labels": sorted(TARGET_LABELS),
    }
    cache: dict = {"settings": settings, "images": {}}
    if cache_path.is_file():
        loaded = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and loaded.get("settings") == settings:
            cache["images"] = loaded.get("images", {})

    unique_paths = list(dict.fromkeys(path.resolve() for path in image_paths))
    pending = []
    for path in unique_paths:
        key = str(path)
        entry = cache["images"].get(key)
        if not isinstance(entry, dict) or entry.get("signature") != _file_signature(path):
            pending.append(path)

    if pending:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "NudeNet .pt weights require Ultralytics: pip install ultralytics"
            ) from exc

        model = YOLO(str(weights))
        missing = TARGET_LABELS - set(model.names.values())
        if missing:
            raise ValueError(f"NudeNet weights lack expected labels: {sorted(missing)}")

        print(f"Detecting NudeNet regions in {len(pending)} clean images...")
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            results = model.predict(
                [str(path) for path in batch],
                imgsz=resolution,
                conf=confidence,
                verbose=False,
            )
            if len(results) != len(batch):
                raise RuntimeError("NudeNet returned an unexpected number of results")

            for path, result in zip(batch, results):
                boxes: list[list[float]] = []
                if result.boxes is not None:
                    for xyxy, class_id in zip(
                        result.boxes.xyxy.cpu().tolist(),
                        result.boxes.cls.cpu().tolist(),
                    ):
                        if result.names[int(class_id)] in TARGET_LABELS:
                            boxes.append([float(value) for value in xyxy])
                cache["images"][str(path)] = {
                    "signature": _file_signature(path),
                    "boxes": boxes,
                }
            _save_cache(cache_path, cache)

    regions = {
        str(path): [tuple(box) for box in cache["images"][str(path)]["boxes"]]
        for path in unique_paths
    }
    positives = sum(bool(boxes) for boxes in regions.values())
    print(f"NudeNet regions: {positives}/{len(regions)} clean images have target detections.")
    return regions
