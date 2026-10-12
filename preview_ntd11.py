"""Save visual checks of the mosaic-tool YOLO segmentation checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Images to inspect.")
    parser.add_argument(
        "--weights", type=Path,
        default=Path("pretrain-weights/ntd11_anime_nsfw_segm_v5-variant1.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("samples/ntd11_inference"))
    parser.add_argument("--nudenet-cache", type=Path, default=Path("data/nudenet_regions.json"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _caption(image: Image.Image, title: str) -> Image.Image:
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, 24), fill=(255, 255, 255))
    draw.text((6, 5), title, fill=(0, 0, 0))
    return panel


def _paint_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    pixels = np.asarray(image).copy()
    pixels[mask] = (pixels[mask] * 0.6 + np.asarray(color) * 0.4).astype(np.uint8)
    return Image.fromarray(pixels)


def _draw_detection(
    image: Image.Image, box: list[float], label: str, color: tuple[int, int, int]
) -> None:
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
    text_y = max(25, int(y1) - 16)
    draw.text((int(x1) + 2, text_y), label, fill=color, stroke_width=2,
              stroke_fill=(255, 255, 255))


def _nudenet_boxes(cache_path: Path) -> dict[str, list[list[float]]]:
    if not cache_path.is_file():
        return {}
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    return {
        str(Path(path).resolve()): entry.get("boxes", [])
        for path, entry in cache.get("images", {}).items()
    }


def _mask_at_image_size(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if mask.shape != (size[1], size[0]):
        mask = np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST))
    return mask > 0.5


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # mosaic-tool uses Ultralytics segmentation with low raw confidence and
    # retina masks, then accepts genital classes at its configured threshold.
    mosaic_config_path = Path(__file__).resolve().parent / "mosaic-tool/config.yaml"
    mosaic_config = yaml.safe_load(mosaic_config_path.read_text(encoding="utf-8"))
    genital_config = mosaic_config["groups"]["genital"]
    genital_classes = {name.lower() for name in genital_config["classes"]}
    genital_confidence = float(genital_config["conf"])
    nudenet_boxes = _nudenet_boxes(args.nudenet_cache)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    if model.task != "segment":
        raise ValueError(f"Expected a YOLO segmentation checkpoint, got {model.task}")
    print(f"Model classes: {model.names}")

    records = []
    overview_rows = []
    for source in args.images:
        if not source.is_file():
            raise FileNotFoundError(source)
        original = Image.open(source).convert("RGB")
        bgr = np.asarray(original)[:, :, ::-1].copy()
        predict_options = {
            "conf": args.confidence,
            "imgsz": args.imgsz,
            "retina_masks": True,
            "verbose": False,
        }
        if args.device is not None:
            predict_options["device"] = args.device
        result = model.predict(bgr, **predict_options)[0]

        source_key = str(source.resolve())
        nude_panel = original.copy()
        for box in nudenet_boxes.get(source_key, []):
            _draw_detection(nude_panel, box, "NudeNet", (0, 125, 255))

        candidates = []
        genital_mask = np.zeros((original.height, original.width), dtype=bool)
        raw_panel = original.copy()
        accepted_panel = original.copy()
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            masks = result.masks.data.cpu().numpy() if result.masks is not None else None
            for index, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
                name = str(result.names[int(class_id)])
                is_genital = name.lower() in genital_classes
                accepted = is_genital and float(score) >= genital_confidence
                color = (225, 45, 45) if is_genital else (30, 105, 225)
                binary_mask = (
                    _mask_at_image_size(masks[index], original.size)
                    if masks is not None else np.zeros_like(genital_mask)
                )
                raw_panel = _paint_mask(raw_panel, binary_mask, color)
                _draw_detection(raw_panel, box.tolist(), f"{name} {score:.3f}", color)
                if accepted:
                    genital_mask |= binary_mask
                    accepted_panel = _paint_mask(accepted_panel, binary_mask, color)
                    _draw_detection(accepted_panel, box.tolist(), f"{name} {score:.3f}", color)
                candidates.append({
                    "class": name,
                    "confidence": float(score),
                    "bbox": [float(value) for value in box],
                    "accepted_genital": accepted,
                    "mask_pixels": int(binary_mask.sum()),
                })

        name = f"{source.parent.name}_{source.stem}"
        panels = [
            _caption(original, "Original"),
            _caption(
                nude_panel,
                f"NudeNet cached boxes: {len(nudenet_boxes[source_key])}"
                if source_key in nudenet_boxes else "NudeNet: not cached",
            ),
            _caption(raw_panel, f"NTD11 candidates: {len(candidates)}"),
            _caption(accepted_panel, f"NTD11 genital >= {genital_confidence:.2f}"),
        ]
        preview = Image.new("RGB", (original.width * len(panels), original.height))
        for column, panel in enumerate(panels):
            preview.paste(panel, (column * original.width, 0))
        preview.save(args.output_dir / f"{name}_preview.png")
        Image.fromarray(genital_mask.astype(np.uint8) * 255).save(
            args.output_dir / f"{name}_genital_mask.png"
        )
        overview_rows.append(preview.resize((1024, 256), Image.Resampling.LANCZOS))
        records.append({
            "source": source_key,
            "preview": f"{name}_preview.png",
            "genital_mask": f"{name}_genital_mask.png",
            "nudenet_box_count": len(nudenet_boxes.get(source_key, [])),
            "detections": candidates,
        })
        print(
            f"{name}: {len(candidates)} candidates, "
            f"{sum(item['accepted_genital'] for item in candidates)} accepted genital"
        )

    overview = Image.new("RGB", (1024, 256 * len(overview_rows)), "white")
    for row, panel in enumerate(overview_rows):
        overview.paste(panel, (0, row * 256))
    overview.save(args.output_dir / "overview.png")
    (args.output_dir / "detections.json").write_text(
        json.dumps({
            "weights": str(args.weights.resolve()),
            "raw_confidence": args.confidence,
            "genital_confidence": genital_confidence,
            "imgsz": args.imgsz,
            "images": records,
        }, indent=2), encoding="utf-8"
    )
    print(f"Saved previews and masks to {args.output_dir}")


if __name__ == "__main__":
    main()
