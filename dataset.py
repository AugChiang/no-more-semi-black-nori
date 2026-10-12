from pathlib import Path
import random
from typing import Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from nudenet_regions import RegionBox
from utils import list_image_paths, pil_to_tensor
from torchvision.transforms import functional as TF


def keep_outermost_boxes(boxes: Sequence[RegionBox]) -> list[RegionBox]:
    """Keep wider enclosing detections while retaining distinct overlapping ones."""
    ordered = sorted(
        boxes,
        key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        reverse=True,
    )
    distinct = []
    for box in ordered:
        if any(
            outer[0] <= box[0] and outer[1] <= box[1]
            and outer[2] >= box[2] and outer[3] >= box[3]
            for outer in distinct
        ):
            continue
        distinct.append(box)
    return distinct


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
            angle = np.deg2rad(random.choice(range(0, 180)))
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
    """Place translucent bar groups inside detected regions only."""

    def __init__(
        self,
        alpha_range: Tuple[float, float] = (0.20, 0.80),
        bar_count: Tuple[int, int] = (1, 1),
        bars_per_group: Tuple[int, int] = (3, 6),
        bar_thickness: Tuple[float, float] = (0.035, 0.055),
        bar_gap: Tuple[float, float] = (0.8, 1.8),
        guided_bar_width: Tuple[float, float] = (0.70, 0.90),
        guided_angle_range: Tuple[float, float] = (-8.0, 8.0),
        guided_center_jitter: float = 0.10,
        guided_bar_shift: float = 0.03,
    ) -> None:
        if not 0.0 < guided_bar_width[0] <= guided_bar_width[1] <= 1.0:
            raise ValueError("guided_bar_width must be within (0, 1] and ordered")
        if not -45.0 < guided_angle_range[0] <= guided_angle_range[1] < 45.0:
            raise ValueError("guided_angle_range must be ordered within (-45, 45) degrees")
        if not 0.0 <= guided_center_jitter <= 1.0:
            raise ValueError("guided_center_jitter must be within [0, 1]")
        if not 0.0 <= guided_bar_shift <= 1.0:
            raise ValueError("guided_bar_shift must be within [0, 1]")
        self.alpha_range = alpha_range
        self.bar_count = bar_count
        self.bars_per_group = bars_per_group
        self.bar_thickness = bar_thickness
        self.bar_gap = bar_gap
        self.guided_bar_width = guided_bar_width
        self.guided_angle_range = guided_angle_range
        self.guided_center_jitter = guided_center_jitter
        self.guided_bar_shift = guided_bar_shift

    def __call__(
        self,
        image: torch.Tensor,
        target_box: Optional[RegionBox] = None,
        target_boxes: Optional[Sequence[RegionBox]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 3:
            raise ValueError("Expected a CHW tensor")
        if target_box is not None and target_boxes is not None:
            raise ValueError("Specify target_box or target_boxes, not both")

        _, h, w = image.shape
        mask = np.zeros((h, w), dtype=np.float32)
        if target_box is not None:
            requested_boxes = (target_box,)
        else:
            requested_boxes = target_boxes if target_boxes is not None else ()
        boxes = []
        for x0, y0, x1, y1 in requested_boxes:
            box = (
                int(np.clip(np.ceil(x0), 0, w)),
                int(np.clip(np.ceil(y0), 0, h)),
                int(np.clip(np.ceil(x1), 0, w)),
                int(np.clip(np.ceil(y1), 0, h)),
            )
            if box[2] > box[0] and box[3] > box[1] and box not in boxes:
                boxes.append(box)

        if boxes:
            self._add_bars(mask, target_boxes=boxes)

        mask = np.clip(mask, 0.0, 0.92)
        mask_t = torch.from_numpy(mask).to(device=image.device, dtype=image.dtype).unsqueeze(0)
        corrupted = image * (1.0 - mask_t)
        return corrupted.clamp(0.0, 1.0), mask_t

    def add_to_numpy(
        self,
        image: np.ndarray,
        target_box: Optional[RegionBox] = None,
        target_boxes: Optional[Sequence[RegionBox]] = None,
    ) -> np.ndarray:
        if image.ndim == 2:
            tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            corrupted, _ = self(tensor, target_box=target_box, target_boxes=target_boxes)
            out = corrupted.squeeze(0).numpy() * 255.0
        elif image.ndim == 3:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            corrupted, _ = self(tensor, target_box=target_box, target_boxes=target_boxes)
            out = corrupted.permute(1, 2, 0).numpy() * 255.0
        else:
            raise ValueError("Expected HxW or HxWxC numpy image")
        return np.clip(out, 0, 255).astype(np.uint8)

    def _alpha(self) -> float:
        lo, hi = self.alpha_range
        return random.uniform(lo, hi)

    def _add_bars(
        self,
        mask: np.ndarray,
        target_boxes: Sequence[tuple[int, int, int, int]] = (),
    ) -> None:
        """Place a separate non-overlapping bar group inside each target box."""
        for box in keep_outermost_boxes(target_boxes):
            group_count = random.randint(*self.bar_count)
            for _ in range(group_count):
                self._add_bar_group(mask, box)

    def _add_bar_group(
        self,
        mask: np.ndarray,
        bounds: tuple[int, int, int, int],
    ) -> bool:
        """Fit parallel bars inside bounds, varying every gap independently."""
        h, w = mask.shape
        x0, y0, x1, y1 = bounds
        roi_w, roi_h = x1 - x0, y1 - y0
        size = min(h, w)
        chosen_count = random.randint(*self.bars_per_group)

        for bars in range(chosen_count, 0, -1):
            for _ in range(30):
                angle = random.uniform(*self.guided_angle_range)
                radians = np.deg2rad(angle)
                along = np.array((np.cos(radians), np.sin(radians)))
                across = np.array((-np.sin(radians), np.cos(radians)))
                thickness = random.uniform(*self.bar_thickness) * size
                gap_ratios = [random.uniform(*self.bar_gap) for _ in range(bars - 1)]
                target_widths = [
                    random.uniform(*self.guided_bar_width) * roi_w
                    for _ in range(bars)
                ]
                tilt_height = max(target_widths) * abs(np.tan(radians))
                available_height = roi_h - 2 - tilt_height
                if available_height <= 0:
                    continue
                thickness = min(
                    thickness,
                    available_height / (bars + sum(gap_ratios) + 0.2),
                )
                if bars > 1 and thickness < max(2.0, size * 0.01):
                    break
                lengths = [
                    (width - thickness * abs(along[1])) / abs(along[0])
                    for width in target_widths
                ]
                if min(lengths) <= 1:
                    continue
                gaps = [max(2.0, ratio * thickness) for ratio in gap_ratios]
                rows = [0.0]
                for gap in gaps:
                    rows.append(rows[-1] + thickness + gap)
                middle = (rows[0] + rows[-1]) / 2
                polygons = []
                for row, length in zip(rows, lengths):
                    shift = random.uniform(
                        -self.guided_bar_shift, self.guided_bar_shift
                    ) * roi_w
                    center = across * (row - middle) + along * shift
                    half_length = length / 2
                    half_thickness = thickness * random.uniform(0.95, 1.05) / 2
                    polygons.append(np.array([
                        center + along * half_length + across * half_thickness,
                        center + along * half_length - across * half_thickness,
                        center - along * half_length - across * half_thickness,
                        center - along * half_length + across * half_thickness,
                    ]))

                points = np.concatenate(polygons)
                span = points.max(axis=0) - points.min(axis=0)
                if span[0] > roi_w - 2 or span[1] > roi_h - 2:
                    continue
                if bars > 1 and thickness < max(2.0, size * 0.01):
                    break
                x_min = int(np.ceil(x0 - points[:, 0].min()))
                x_max = int(np.floor(x1 - 1 - points[:, 0].max()))
                y_min = int(np.ceil(y0 - points[:, 1].min()))
                y_max = int(np.floor(y1 - 1 - points[:, 1].max()))
                if x_min > x_max or y_min > y_max:
                    continue

                origin = np.array((
                    int(np.clip(
                        (x0 + x1) / 2 + random.uniform(
                            -self.guided_center_jitter,
                            self.guided_center_jitter,
                        ) * roi_w,
                        x_min, x_max,
                    )),
                    int(np.clip(
                        (y0 + y1) / 2 + random.uniform(
                            -self.guided_center_jitter,
                            self.guided_center_jitter,
                        ) * roi_h,
                        y_min, y_max,
                    )),
                ))
                layer = np.zeros((h, w), dtype=np.uint8)
                for polygon in polygons:
                    corners = np.rint(polygon + origin).astype(np.int32)
                    cv2.fillConvexPoly(layer, corners, 255, lineType=cv2.LINE_8)

                occupied = layer > 0
                if cv2.connectedComponents(layer)[0] - 1 != bars:
                    continue
                ys, xs = np.where(occupied)
                if (
                    not len(xs) or xs.min() < x0 or xs.max() >= x1
                    or ys.min() < y0 or ys.max() >= y1
                ):
                    continue
                existing = cv2.dilate(
                    (mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8)
                )
                if np.any(occupied & (existing > 0)):
                    continue
                np.maximum(mask, layer.astype(np.float32) * (self._alpha() / 255), out=mask)
                return True
        return False


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
        screentone_synthesizer: Optional[ScreentoneSynthesizer] = None,
        return_mask: bool = False,
        image_paths: Optional[Sequence[Union[str, Path]]] = None,
        regions_by_path: Optional[Mapping[str, Sequence[RegionBox]]] = None,
        region_crop_probability: float = 1.0,
        validation_seed: int = 0,
    ) -> None:
        self.paths = (
            [Path(path) for path in image_paths]
            if image_paths is not None
            else list_image_paths(image_path)
        )
        if not self.paths:
            raise ValueError("RestorationDataset requires at least one image")
        if not 0.0 <= region_crop_probability <= 1.0:
            raise ValueError("region_crop_probability must be between 0 and 1")
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.training = training
        self.color_mode = color_mode
        self.return_mask = return_mask
        self.regions_by_path = regions_by_path or {}
        self.region_crop_probability = region_crop_probability
        self.validation_seed = validation_seed
        self.augmentor = augmentor or BlackArtifactAugmentor()
        self.screentone_synth = screentone_synthesizer or ScreentoneSynthesizer(
            probability=screentone_probability
        )

    def __len__(self) -> int:
        if self.num_patches is not None:
            return self.num_patches
        return len(self.paths)

    def __getitem__(self, index: int):
        if self.training:
            return self._get_item(index)

        # Stable corruption per validation image without caching all image tensors.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            random.seed(self.validation_seed + index)
            np.random.seed((self.validation_seed + index) % (2**32))
            return self._get_item(index)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def _get_item(self, index: int):
        if self.training and self.num_patches is not None:
            path = random.choice(self.paths)
        else:
            path = self.paths[index % len(self.paths)]
        regions = keep_outermost_boxes(
            self.regions_by_path.get(str(path.resolve()), ())
        )
        focus_box = None
        if regions:
            focus_box = random.choice(regions) if self.training else regions[0]
        image = Image.open(path)
        image, local_boxes = self._prepare_patch_with_regions(image, regions, focus_box)
        target = pil_to_tensor(image, self.color_mode)
        if self.training:
            target = self.screentone_synth(target)
        corrupted, mask = self.augmentor(target, target_boxes=local_boxes)
        if self.return_mask:
            return corrupted, target, mask
        return corrupted, target

    def _prepare_patch(
        self,
        image: Image.Image,
        focus_box: Optional[RegionBox],
    ) -> tuple[Image.Image, Optional[RegionBox]]:
        """Crop and transform a single box for callers using the older helper."""
        boxes = [focus_box] if focus_box is not None else []
        patch, local_boxes = self._prepare_patch_with_regions(image, boxes, focus_box)
        return patch, local_boxes[0] if local_boxes else None

    def _prepare_patch_with_regions(
        self,
        image: Image.Image,
        regions: Sequence[RegionBox],
        focus_box: Optional[RegionBox],
    ) -> tuple[Image.Image, list[RegionBox]]:
        """Crop around one detection and transform every visible box."""
        original_w, original_h = image.size
        size = self.patch_size
        scale = max(size / min(original_w, original_h), 1.0)
        if scale > 1.0:
            image = image.resize(
                (int(round(original_w * scale)), int(round(original_h * scale))),
                Image.BICUBIC,
            )
        w, h = image.size
        sx, sy = w / original_w, h / original_h
        scaled_regions = [
            (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
            for box in regions
        ]
        if focus_box is not None:
            focus_box = (
                focus_box[0] * sx,
                focus_box[1] * sy,
                focus_box[2] * sx,
                focus_box[3] * sy,
            )

        use_focus = focus_box is not None and (
            not self.training or random.random() < self.region_crop_probability
        )
        if use_focus:
            center_x = (focus_box[0] + focus_box[2]) / 2
            center_y = (focus_box[1] + focus_box[3]) / 2
            if self.training:
                center_x += random.uniform(-0.15, 0.15) * size
                center_y += random.uniform(-0.15, 0.15) * size
            left = int(np.clip(round(center_x - size / 2), 0, w - size))
            top = int(np.clip(round(center_y - size / 2), 0, h - size))
        elif self.training:
            left = random.randint(0, w - size)
            top = random.randint(0, h - size)
        else:
            left = (w - size) // 2
            top = (h - size) // 2

        image = image.crop((left, top, left + size, top + size))
        local_boxes = []
        for box in scaled_regions:
            x0 = max(0.0, box[0] - left)
            y0 = max(0.0, box[1] - top)
            x1 = min(float(size), box[2] - left)
            y1 = min(float(size), box[3] - top)
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                local_boxes.append((x0, y0, x1, y1))

        if self.training and random.random() < 0.5:
            image = TF.hflip(image)
            local_boxes = [
                (size - x1, y0, size - x0, y1)
                for x0, y0, x1, y1 in local_boxes
            ]
        if self.training and random.random() < 0.5:
            image = TF.vflip(image)
            local_boxes = [
                (x0, size - y1, x1, size - y0)
                for x0, y0, x1, y1 in local_boxes
            ]
        return image, local_boxes
