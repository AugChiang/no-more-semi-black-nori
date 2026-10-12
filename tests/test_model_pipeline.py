import random
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from dataset import RestorationDataset
from infer import load_checkpoint
from model import DualDomainNAFNet


class ModelPipelineTests(unittest.TestCase):
    def test_nafnet_preserves_odd_shapes_in_gray_and_rgb(self) -> None:
        for channels in (1, 3):
            with self.subTest(channels=channels):
                model = DualDomainNAFNet(
                    img_channel=channels,
                    width=8,
                    middle_blk_num=1,
                    enc_blk_nums=(1, 1),
                    dec_blk_nums=(1, 1),
                )
                image = torch.rand(1, channels, 31, 47)
                output = model(image)
                self.assertEqual(output.shape, image.shape)
                self.assertTrue(torch.allclose(output, image))

    def test_checkpoint_loader_uses_saved_stages_and_legacy_fft(self) -> None:
        for use_fft, stages in ((False, (1, 1)), (True, (1, 1, 1, 1))):
            with self.subTest(use_fft=use_fft):
                model = DualDomainNAFNet(
                    width=8,
                    middle_blk_num=1,
                    enc_blk_nums=stages,
                    dec_blk_nums=stages,
                    use_fft=use_fft,
                )
                metadata = {
                    "img_channel": 1,
                    "width": 8,
                    "middle_blk_num": 1,
                }
                if not use_fft:
                    metadata.update({
                        "enc_blk_nums": list(stages),
                        "dec_blk_nums": list(stages),
                        "use_fft": False,
                    })
                checkpoint = {"model": model.state_dict(), "model_args": metadata}
                with patch("infer.torch.load", return_value=checkpoint):
                    loaded = load_checkpoint(
                        Path("unused.pth"), torch.device("cpu"), 8, 1
                    )
                self.assertEqual(loaded.use_fft, use_fft)
                self.assertEqual(loaded.enc_blk_nums, stages)

    def test_validation_corruption_repeats_without_changing_random_state(self) -> None:
        path = Path("validation_test.png")
        dataset = RestorationDataset(
            path,
            image_paths=[path],
            patch_size=64,
            training=False,
            return_mask=True,
            validation_seed=73,
        )
        random.seed(123)
        state = random.getstate()
        with patch("dataset.Image.open", return_value=Image.new("L", (64, 64), 255)):
            first = dataset[0]
            second = dataset[0]
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertEqual(random.getstate(), state)


if __name__ == "__main__":
    unittest.main()
