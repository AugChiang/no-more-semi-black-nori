import random
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image

from dataset import BlackArtifactAugmentor, RestorationDataset


class BlackArtifactAugmentorTests(unittest.TestCase):
    def test_guided_masks_are_sparse_separate_bar_groups(self) -> None:
        augmentor = BlackArtifactAugmentor()

        for size in (64, 256, 500):
            for seed in range(100):
                with self.subTest(size=size, seed=seed):
                    random.seed(seed)
                    _, mask = augmentor(
                        torch.ones(1, size, size),
                        target_box=(0.0, 0.0, float(size), float(size)),
                    )
                    binary = (mask[0].numpy() > 0.01).astype(np.uint8)
                    components, _ = cv2.connectedComponents(binary)

                    self.assertGreaterEqual(components - 1, 3)
                    self.assertLessEqual(components - 1, 6)
                    self.assertGreater(binary.mean(), 0.01)
                    self.assertLess(binary.mean(), 0.35)

    def test_no_detection_leaves_image_unmasked(self) -> None:
        image = torch.rand(1, 128, 128)
        corrupted, mask = BlackArtifactAugmentor()(image)
        self.assertTrue(torch.equal(corrupted, image))
        self.assertEqual(int(torch.count_nonzero(mask)), 0)

    def test_mask_blends_gray_and_rgb_inputs(self) -> None:
        augmentor = BlackArtifactAugmentor()

        for channels in (1, 3):
            with self.subTest(channels=channels):
                random.seed(42)
                image = torch.rand(channels, 256, 256)
                corrupted, mask = augmentor(
                    image, target_boxes=[(40.0, 40.0, 200.0, 200.0)]
                )

                self.assertEqual(corrupted.shape, image.shape)
                self.assertEqual(mask.shape, (1, 256, 256))
                self.assertTrue(torch.allclose(corrupted, image * (1 - mask)))
                self.assertGreaterEqual(float(mask.max()), 0.20)
                self.assertLessEqual(float(mask.max()), 0.80)

    def test_guided_bars_stay_inside_target_region(self) -> None:
        augmentor = BlackArtifactAugmentor()
        box = (110.0, 105.0, 145.0, 140.0)

        for seed in range(100):
            with self.subTest(seed=seed):
                random.seed(seed)
                _, mask = augmentor(torch.ones(1, 256, 256), target_box=box)
                coverage = (mask[0, 105:140, 110:145] > 0).float().mean()
                self.assertGreater(float(coverage), 0.08)
                outside = mask[0].clone()
                outside[105:140, 110:145] = 0
                self.assertEqual(int(torch.count_nonzero(outside)), 0)

    def test_all_detected_boxes_receive_bars_without_spill(self) -> None:
        augmentor = BlackArtifactAugmentor()
        boxes = (
            (2.1, 181.3, 266.3, 511.0),
            (206.5, 150.4, 320.7, 284.1),
        )
        allowed = np.zeros((512, 512), dtype=bool)
        for x0, y0, x1, y1 in boxes:
            allowed[int(np.ceil(y0)):int(np.ceil(y1)),
                    int(np.ceil(x0)):int(np.ceil(x1))] = True

        for seed in range(50):
            with self.subTest(seed=seed):
                random.seed(seed)
                _, mask = augmentor(torch.ones(1, 512, 512), target_boxes=boxes)
                binary = mask[0].numpy() > 0
                self.assertFalse(np.any(binary & ~allowed))
                for x0, y0, x1, y1 in boxes:
                    self.assertTrue(binary[
                        int(np.ceil(y0)):int(np.ceil(y1)),
                        int(np.ceil(x0)):int(np.ceil(x1)),
                    ].any())

    def test_nested_detections_use_wider_box(self) -> None:
        augmentor = BlackArtifactAugmentor()
        outer = (101.8, 27.8, 252.4, 226.7)
        inner = (113.0, 61.5, 241.3, 215.1)
        for seed in range(20):
            with self.subTest(seed=seed):
                random.seed(seed)
                _, nested_mask = augmentor(
                    torch.ones(1, 512, 512), target_boxes=[outer, inner]
                )
                random.seed(seed)
                _, outer_mask = augmentor(torch.ones(1, 512, 512), target_box=outer)
                self.assertTrue(torch.equal(nested_mask, outer_mask))

    def test_guided_width_follows_configured_box_fraction(self) -> None:
        box = (80.0, 50.0, 280.0, 230.0)
        for width_range in ((0.70, 0.90), (0.50, 0.60)):
            augmentor = BlackArtifactAugmentor(guided_bar_width=width_range)
            for seed in range(30):
                with self.subTest(width_range=width_range, seed=seed):
                    random.seed(seed)
                    _, mask = augmentor(torch.ones(1, 512, 512), target_box=box)
                    binary = (mask[0].numpy() > 0).astype(np.uint8)
                    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
                    self.assertGreater(count, 1)
                    widths = stats[1:, cv2.CC_STAT_WIDTH]
                    self.assertGreaterEqual(widths.min(), width_range[0] * 200 - 2)
                    self.assertLessEqual(widths.max(), width_range[1] * 200 + 2)

    def test_bar_gaps_can_vary_within_a_group(self) -> None:
        augmentor = BlackArtifactAugmentor(
            bars_per_group=(6, 6), guided_angle_range=(0.0, 0.0),
        )
        found_uneven_gaps = False
        for seed in range(20):
            random.seed(seed)
            _, mask = augmentor(
                torch.ones(1, 256, 256), target_box=(10.0, 10.0, 246.0, 246.0)
            )
            binary = (mask[0].numpy() > 0).astype(np.uint8)
            count, _, _, centroids = cv2.connectedComponentsWithStats(binary)
            self.assertEqual(count - 1, 6)
            positions = sorted(centroids[1:, 1])
            distances = np.diff(positions)
            if np.ptp(distances) > 2.0:
                found_uneven_gaps = True
        self.assertTrue(found_uneven_gaps)

    def test_guided_angle_range_controls_bar_angle(self) -> None:
        for angle in (0.0, 20.0):
            with self.subTest(angle=angle):
                augmentor = BlackArtifactAugmentor(
                    bars_per_group=(1, 1),
                    guided_angle_range=(angle, angle),
                    guided_center_jitter=0.0,
                    guided_bar_shift=0.0,
                )
                random.seed(7)
                _, mask = augmentor(
                    torch.ones(1, 256, 256),
                    target_box=(20.0, 20.0, 236.0, 236.0),
                )
                ys, xs = np.nonzero(mask[0].numpy())
                self.assertGreater(len(xs), 0)
                measured = np.degrees(np.arctan(np.polyfit(xs, ys, 1)[0]))
                self.assertAlmostEqual(measured, angle, delta=3.0)

    def test_region_box_tracks_crop_and_flips(self) -> None:
        pixels = np.zeros((512, 512), dtype=np.uint8)
        pixels[300:330, 200:230] = 255
        dataset = RestorationDataset(
            "unused.png",
            image_paths=[Path("unused.png")],
            patch_size=128,
            region_crop_probability=1.0,
        )

        for seed in range(20):
            with self.subTest(seed=seed):
                random.seed(seed)
                patch, box = dataset._prepare_patch(
                    Image.fromarray(pixels), (200.0, 300.0, 230.0, 330.0)
                )
                self.assertIsNotNone(box)
                ys, xs = np.where(np.asarray(patch) > 0)
                self.assertEqual(box, (float(xs.min()), float(ys.min()),
                                       float(xs.max() + 1), float(ys.max() + 1)))

    def test_crop_transforms_every_visible_region(self) -> None:
        pixels = np.zeros((256, 256), dtype=np.uint8)
        pixels[100:120, 105:125] = 255
        pixels[130:150, 135:155] = 255
        boxes = [(105.0, 100.0, 125.0, 120.0), (135.0, 130.0, 155.0, 150.0)]
        dataset = RestorationDataset(
            "unused.png",
            image_paths=[Path("unused.png")],
            patch_size=128,
            region_crop_probability=1.0,
        )
        for seed in range(20):
            with self.subTest(seed=seed):
                random.seed(seed)
                patch, local_boxes = dataset._prepare_patch_with_regions(
                    Image.fromarray(pixels), boxes, boxes[0]
                )
                self.assertEqual(len(local_boxes), 2)
                cropped = np.asarray(patch)
                for x0, y0, x1, y1 in local_boxes:
                    self.assertTrue(cropped[
                        int(y0):int(y1), int(x0):int(x1)
                    ].any())

    def test_dataset_passes_wider_nested_box_to_augmentor(self) -> None:
        captured = []

        class RecordingAugmentor(BlackArtifactAugmentor):
            def __call__(self, image, target_box=None, target_boxes=None):
                captured.extend(target_boxes or ())
                return super().__call__(
                    image, target_box=target_box, target_boxes=target_boxes
                )

        outer = (100.0, 40.0, 250.0, 230.0)
        inner = (115.0, 70.0, 235.0, 210.0)
        path = Path("nested_test.png")
        dataset = RestorationDataset(
            path,
            image_paths=[path],
            patch_size=256,
            training=False,
            augmentor=RecordingAugmentor(),
            regions_by_path={str(path.resolve()): [inner, outer]},
        )
        with patch("dataset.Image.open", return_value=Image.new("L", (512, 512), 255)):
            dataset[0]
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][2] - captured[0][0], 150.0)


if __name__ == "__main__":
    unittest.main()
