"""Load official NAFNet restoration weights for fine-tuning."""

from pathlib import Path

import torch

from model import DualDomainNAFNet


def load_nafnet_pretrained(model: DualDomainNAFNet, path: Path) -> None:
    """Load official BasicSR NAFNet weights, adapting RGB weights to grayscale."""
    if not path.is_file():
        raise FileNotFoundError(f"NAFNet pre-trained weights not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported NAFNet checkpoint format: {path}")
    source = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    if not isinstance(source, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in source.items()
    ):
        raise ValueError(f"Unsupported NAFNet state dict: {path}")

    # The official implementation stores middle_blks as one Sequential. This
    # repo stores that Sequential as element zero of a ModuleList.
    state = {
        key.replace("middle_blks.", "middle_blks.0.", 1)
        if key.startswith("middle_blks.") else key: value
        for key, value in source.items()
    }
    if "intro.weight" not in state or "ending.weight" not in state or "ending.bias" not in state:
        raise ValueError(f"NAFNet checkpoint is missing input or output layers: {path}")
    if model.img_channel == 1 and state["intro.weight"].shape[1] == 3:
        # An RGB model receiving three copies of a gray image has this exact
        # first-layer response. Luminance-weighting the RGB residual gives a
        # one-channel output while retaining all interior pretrained layers.
        state["intro.weight"] = state["intro.weight"].sum(dim=1, keepdim=True)
        luminance = state["ending.weight"].new_tensor((0.299, 0.587, 0.114))
        state["ending.weight"] = (
            state["ending.weight"] * luminance[:, None, None, None]
        ).sum(dim=0, keepdim=True)
        state["ending.bias"] = (state["ending.bias"] * luminance).sum().reshape(1)

    target = model.state_dict()
    missing = target.keys() - state.keys()
    unexpected = state.keys() - target.keys()
    mismatched = [
        key for key in target.keys() & state.keys()
        if target[key].shape != state[key].shape
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            f"NAFNet weights do not match the configured model: "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(mismatched)} shape mismatches. Use the official SIDD-width32 "
            "layout (width 32, encoder [2,2,4,8], middle 12, "
            "decoder [2,2,2,2]), or disable pre-training."
        )
    model.load_state_dict(state, strict=True)
