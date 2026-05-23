"""DenseNet-BC for CIFAR-10."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class DenseLayer(nn.Module):
    """One DenseNet-BC layer: BN-ReLU-1x1Conv-BN-ReLU-3x3Conv."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        bottleneck: bool = True,
    ) -> None:
        super().__init__()
        self.drop_rate = drop_rate
        self.bottleneck = bottleneck

        if bottleneck:
            inter_channels = bn_size * growth_rate
            self.net = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.net = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        bottleneck: bool,
    ) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth_rate, bn_size, drop_rate, bottleneck))
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Transition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseNetCIFAR(nn.Module):
    """CIFAR DenseNet / DenseNet-BC.

    For bottleneck=True, depth follows L = 6n + 4.
    For bottleneck=False, depth follows L = 3n + 4.
    """

    def __init__(
        self,
        depth: int = 100,
        growth_rate: int = 24,
        compression: float = 0.5,
        num_classes: int = 10,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        bottleneck: bool = True,
    ) -> None:
        super().__init__()
        if not 0 < compression <= 1:
            raise ValueError("compression must be in (0, 1]")
        if bottleneck:
            if (depth - 4) % 6 != 0:
                raise ValueError("DenseNet-BC depth should satisfy depth = 6n + 4")
            layers_per_block = (depth - 4) // 6
        else:
            if (depth - 4) % 3 != 0:
                raise ValueError("DenseNet depth should satisfy depth = 3n + 4")
            layers_per_block = (depth - 4) // 3

        self.depth = depth
        self.growth_rate = growth_rate
        self.compression = compression
        self.bottleneck = bottleneck
        self.layers_per_block = layers_per_block

        channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = DenseBlock(
            layers_per_block,
            channels,
            growth_rate,
            bn_size,
            drop_rate,
            bottleneck,
        )
        channels = self.block1.out_channels
        out_channels = math.floor(channels * compression)
        self.trans1 = Transition(channels, out_channels)
        channels = out_channels

        self.block2 = DenseBlock(
            layers_per_block,
            channels,
            growth_rate,
            bn_size,
            drop_rate,
            bottleneck,
        )
        channels = self.block2.out_channels
        out_channels = math.floor(channels * compression)
        self.trans2 = Transition(channels, out_channels)
        channels = out_channels

        self.block3 = DenseBlock(
            layers_per_block,
            channels,
            growth_rate,
            bn_size,
            drop_rate,
            bottleneck,
        )
        channels = self.block3.out_channels

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, num_classes)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.block3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)


def densenet_bc(
    depth: int = 100,
    growth_rate: int = 24,
    compression: float = 0.5,
    num_classes: int = 10,
    drop_rate: float = 0.0,
) -> DenseNetCIFAR:
    return DenseNetCIFAR(
        depth=depth,
        growth_rate=growth_rate,
        compression=compression,
        num_classes=num_classes,
        drop_rate=drop_rate,
        bottleneck=True,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def describe_model(model: nn.Module) -> dict[str, int | float | bool | str]:
    return {
        "model": model.__class__.__name__,
        "depth": getattr(model, "depth", "unknown"),
        "growth_rate": getattr(model, "growth_rate", "unknown"),
        "compression": getattr(model, "compression", "unknown"),
        "bottleneck": getattr(model, "bottleneck", "unknown"),
        "layers_per_block": getattr(model, "layers_per_block", "unknown"),
        "trainable_parameters": count_parameters(model),
    }


if __name__ == "__main__":
    net = densenet_bc()
    dummy = torch.randn(2, 3, 32, 32)
    logits = net(dummy)
    print(net)
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Trainable parameters: {count_parameters(net):,}")
