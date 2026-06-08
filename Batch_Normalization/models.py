"""
VGG-A models used for the Batch Normalization experiment.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


VGG_A_CFG: list[int | str] = [
    64,
    "M",
    128,
    "M",
    256,
    256,
    "M",
    512,
    512,
    "M",
    512,
    512,
    "M",
]


def init_weights_(module: nn.Module) -> None:
    """Initialize modules in the same spirit as the provided project code."""
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def get_number_of_parameters(model: nn.Module) -> int:
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += int(np.prod(parameter.shape).item())
    return parameters_n


def _make_layers(
    cfg: Iterable[int | str],
    inp_ch: int = 3,
    batch_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = inp_ch

    for item in cfg:
        if item == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        out_channels = int(item)
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=not batch_norm,
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels

    return nn.Sequential(*layers)


class _VGGA(nn.Module):
    """VGG-A adapted to 32x32 CIFAR-10 images."""

    def __init__(
        self,
        inp_ch: int = 3,
        num_classes: int = 10,
        batch_norm: bool = False,
        init_weights: bool = True,
    ) -> None:
        super().__init__()

        self.features = _make_layers(VGG_A_CFG, inp_ch=inp_ch, batch_norm=batch_norm)
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


class VGG_A(_VGGA):
    """Baseline VGG-A without Batch Normalization."""

    def __init__(
        self,
        inp_ch: int = 3,
        num_classes: int = 10,
        init_weights: bool = True,
    ) -> None:
        super().__init__(
            inp_ch=inp_ch,
            num_classes=num_classes,
            batch_norm=False,
            init_weights=init_weights,
        )


class VGG_A_BatchNorm(_VGGA):
    """VGG-A with BatchNorm2d after each convolution layer."""

    def __init__(
        self,
        inp_ch: int = 3,
        num_classes: int = 10,
        init_weights: bool = True,
    ) -> None:
        super().__init__(
            inp_ch=inp_ch,
            num_classes=num_classes,
            batch_norm=True,
            init_weights=init_weights,
        )


if __name__ == "__main__":
    for model_cls in (VGG_A, VGG_A_BatchNorm):
        model = model_cls()
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        print(model_cls.__name__, y.shape, get_number_of_parameters(model))
