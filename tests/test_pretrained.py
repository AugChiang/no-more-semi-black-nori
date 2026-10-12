import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from model import DualDomainNAFNet
from pretrained import load_nafnet_pretrained


class PretrainedWeightsTests(unittest.TestCase):
    def test_official_rgb_weights_transfer_to_grayscale(self) -> None:
        torch.manual_seed(17)
        rgb_model = DualDomainNAFNet(
            img_channel=3,
            width=4,
            middle_blk_num=1,
            enc_blk_nums=(1, 1),
            dec_blk_nums=(1, 1),
        ).eval()
        with torch.no_grad():
            rgb_model.ending.weight.normal_(std=0.01)
            rgb_model.ending.bias.normal_(std=0.01)

        official_state = {
            key.replace("middle_blks.0.", "middle_blks.", 1)
            if key.startswith("middle_blks.0.") else key: value.clone()
            for key, value in rgb_model.state_dict().items()
        }
        gray_model = DualDomainNAFNet(
            img_channel=1,
            width=4,
            middle_blk_num=1,
            enc_blk_nums=(1, 1),
            dec_blk_nums=(1, 1),
        ).eval()
        path = Path("official.pth")
        with patch("pretrained.Path.is_file", return_value=True), patch(
            "pretrained.torch.load", return_value={"params": official_state}
        ):
            load_nafnet_pretrained(gray_model, path)

            image = torch.rand(1, 1, 19, 23)
            luminance = image.new_tensor((0.299, 0.587, 0.114))
            with torch.inference_mode():
                expected = (
                    rgb_model(image.repeat(1, 3, 1, 1))
                    * luminance[None, :, None, None]
                ).sum(dim=1, keepdim=True)
                actual = gray_model(image)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

            incompatible = DualDomainNAFNet(
                img_channel=1,
                width=4,
                middle_blk_num=2,
                enc_blk_nums=(1, 1),
                dec_blk_nums=(1, 1),
            )
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_nafnet_pretrained(incompatible, path)


if __name__ == "__main__":
    unittest.main()
