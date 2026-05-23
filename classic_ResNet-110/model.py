"""ResNet-110 for CIFAR-10.

This is the classic CIFAR-style ResNet from "Deep Residual Learning for
Image Recognition": depth = 6n + 2, with n = 18 for ResNet-110.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class IdentityPadding(nn.Module):
    """Option A shortcut used by the original CIFAR ResNet paper."""

    def __init__(self, in_planes: int, out_planes: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.pad_channels = out_planes - in_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride != 1:
            x = x[:, :, :: self.stride, :: self.stride]
        if self.pad_channels <= 0:
            return x
        pad_front = self.pad_channels // 2
        pad_back = self.pad_channels - pad_front
        return F.pad(x, (0, 0, 0, 0, pad_front, pad_back), "constant", 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        shortcut_type: str = "A",
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1 and in_planes == planes:
            self.shortcut = nn.Identity()
        elif shortcut_type.upper() == "A":
            self.shortcut = IdentityPadding(in_planes, planes, stride)
        elif shortcut_type.upper() == "B":
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            raise ValueError("shortcut_type must be 'A' or 'B'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNetCIFAR(nn.Module):
    """CIFAR ResNet with 3 stages and 6n+2 layers."""

    def __init__(
        self,
        depth: int = 110,
        num_classes: int = 10,
        block: type[BasicBlock] = BasicBlock,
        shortcut_type: str = "A",
    ) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("CIFAR ResNet depth should satisfy depth = 6n + 2")

        n_blocks = (depth - 2) // 6
        self.depth = depth
        self.in_planes = 16

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n_blocks, stride=1, shortcut_type=shortcut_type)
        self.layer2 = self._make_layer(block, 32, n_blocks, stride=2, shortcut_type=shortcut_type)
        self.layer3 = self._make_layer(block, 64, n_blocks, stride=2, shortcut_type=shortcut_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int,
        shortcut_type: str,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(block(self.in_planes, planes, stride, shortcut_type))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, 1, shortcut_type))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
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
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)


def resnet110(num_classes: int = 10, shortcut_type: str = "A") -> ResNetCIFAR:
    return ResNetCIFAR(depth=110, num_classes=num_classes, shortcut_type=shortcut_type)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def describe_model(model: nn.Module) -> dict[str, int | str]:
    return {
        "model": model.__class__.__name__,
        "depth": getattr(model, "depth", "unknown"),
        "trainable_parameters": count_parameters(model),
    }


if __name__ == "__main__":
    net = resnet110()
    dummy = torch.randn(2, 3, 32, 32)
    logits = net(dummy)
    print(net)
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Trainable parameters: {count_parameters(net):,}")
