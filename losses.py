import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


class CharbonnierLoss(nn.Module):
    """Smooth reconstruction loss that remains stable near zero error."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((output - target) ** 2 + self.eps**2))

def masked_charbonnier_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Measure restoration error primarily where synthetic damage was applied."""
    error = torch.sqrt((output - target) ** 2 + eps**2)
    weights = mask.expand_as(error)
    return torch.sum(error * weights) / weights.sum().clamp_min(1.0)


def _channelwise_kernel(kernel: torch.Tensor, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return kernel.to(device=device, dtype=dtype).view(1, 1, *kernel.shape).repeat(channels, 1, 1, 1)


def gradient_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    channels = output.shape[1]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    wx = _channelwise_kernel(kx, channels, output.device, output.dtype)
    wy = _channelwise_kernel(ky, channels, output.device, output.dtype)
    out_x = F.conv2d(output, wx, padding=1, groups=channels)
    out_y = F.conv2d(output, wy, padding=1, groups=channels)
    tgt_x = F.conv2d(target, wx, padding=1, groups=channels)
    tgt_y = F.conv2d(target, wy, padding=1, groups=channels)
    return F.l1_loss(out_x, tgt_x) + F.l1_loss(out_y, tgt_y)


def laplacian_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    channels = output.shape[1]
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    weight = _channelwise_kernel(kernel, channels, output.device, output.dtype)
    out_lap = F.conv2d(output, weight, padding=1, groups=channels)
    tgt_lap = F.conv2d(target, weight, padding=1, groups=channels)
    return F.l1_loss(out_lap, tgt_lap)


def frequency_l1_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    output_fft = torch.fft.rfft2(output, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    out_mag = torch.log1p(torch.abs(output_fft))
    tgt_mag = torch.log1p(torch.abs(target_fft))

    _, _, h, fw = out_mag.shape
    yy = torch.linspace(0.0, 1.0, h, device=output.device, dtype=output.dtype).view(1, 1, h, 1)
    xx = torch.linspace(0.0, 1.0, fw, device=output.device, dtype=output.dtype).view(1, 1, 1, fw)
    high_freq_weight = 0.35 + torch.sqrt(xx**2 + yy**2).clamp(0.0, 1.0)
    return torch.mean(torch.abs(out_mag - tgt_mag) * high_freq_weight)


def local_contrast_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out_mean = F.avg_pool2d(output, kernel_size=9, stride=1, padding=4)
    tgt_mean = F.avg_pool2d(target, kernel_size=9, stride=1, padding=4)
    out_detail = output - out_mean
    tgt_detail = target - tgt_mean
    return F.l1_loss(out_detail, tgt_detail)


def restoration_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    artifact_mask: torch.Tensor,
    criterion: CharbonnierLoss,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the weighted restoration objective and its components."""
    components = {
        "pixel": criterion(output, target),
        "artifact": masked_charbonnier_loss(output, target, artifact_mask),
        "gradient": gradient_loss(output, target),
        "laplacian": laplacian_loss(output, target),
        "frequency": frequency_l1_loss(output, target),
        "contrast": local_contrast_loss(output, target),
    }
    total = (
        components["pixel"]
        + args.artifact_weight * components["artifact"]
        + args.gradient_weight * components["gradient"]
        + args.laplacian_weight * components["laplacian"]
        + args.freq_weight * components["frequency"]
        + args.contrast_weight * components["contrast"]
    )
    return total, components