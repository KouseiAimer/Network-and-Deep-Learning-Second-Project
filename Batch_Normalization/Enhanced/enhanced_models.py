"""
Model variants for extended Batch Normalization experiments.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

BN_ROOT = Path(__file__).resolve().parent.parent
if str(BN_ROOT) not in sys.path:
    sys.path.insert(0, str(BN_ROOT))

from models import VGG_A_CFG, get_number_of_parameters, init_weights_


MODEL_DISPLAY_NAMES = {
    "no_bn": "VGG-A without BN",
    "bn": "Conv-BN-ReLU",
    "bn_after_relu": "Conv-ReLU-BN",
    "bn_first_half": "BN in first half",
    "bn_second_half": "BN in second half",
    "groupnorm": "GroupNorm",
}


def _count_convs(cfg: Iterable[int | str]) -> int:
    return sum(1 for item in cfg if item != "M")


def _valid_group_count(num_channels: int, requested_groups: int) -> int:
    groups = min(num_channels, requested_groups)
    while num_channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def _should_add_norm(mode: str, conv_index: int, total_convs: int) -> bool:
    if mode in {"bn", "bn_after_relu", "groupnorm"}:
        return True
    if mode == "bn_first_half":
        return conv_index <= total_convs // 2
    if mode == "bn_second_half":
        return conv_index > total_convs // 2
    if mode == "no_bn":
        return False
    raise ValueError(f"Unknown normalization mode: {mode}")


def _make_norm_layer(mode: str, channels: int, group_norm_groups: int) -> nn.Module:
    if mode == "groupnorm":
        groups = _valid_group_count(channels, group_norm_groups)
        return nn.GroupNorm(groups, channels)
    return nn.BatchNorm2d(channels)


def _make_layers(
    cfg: Iterable[int | str],
    mode: str,
    inp_ch: int = 3,
    group_norm_groups: int = 32,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = inp_ch
    conv_index = 0
    total_convs = _count_convs(cfg)

    for item in cfg:
        if item == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        conv_index += 1
        out_channels = int(item)
        add_norm = _should_add_norm(mode, conv_index, total_convs)
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=not add_norm,
            )
        )

        if mode == "bn_after_relu" and add_norm:
            layers.append(nn.ReLU(inplace=True))
            layers.append(_make_norm_layer(mode, out_channels, group_norm_groups))
        else:
            if add_norm:
                layers.append(_make_norm_layer(mode, out_channels, group_norm_groups))
            layers.append(nn.ReLU(inplace=True))

        in_channels = out_channels

    return nn.Sequential(*layers)


class EnhancedVGG_A(nn.Module):
    """VGG-A with configurable normalization placement."""

    def __init__(
        self,
        mode: str,
        inp_ch: int = 3,
        num_classes: int = 10,
        group_norm_groups: int = 32,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.features = _make_layers(
            VGG_A_CFG,
            mode=mode,
            inp_ch=inp_ch,
            group_norm_groups=group_norm_groups,
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self.apply(init_weights_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_enhanced_model(model_name: str, group_norm_groups: int = 32) -> nn.Module:
    if model_name not in MODEL_DISPLAY_NAMES:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {sorted(MODEL_DISPLAY_NAMES)}"
        )
    return EnhancedVGG_A(mode=model_name, group_norm_groups=group_norm_groups)


if __name__ == "__main__":
    for name in MODEL_DISPLAY_NAMES:
        model = build_enhanced_model(name)
        output = model(torch.randn(2, 3, 32, 32))
        print(name, output.shape, get_number_of_parameters(model))
