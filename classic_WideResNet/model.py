"""WideResNet for CIFAR-10.

Default model: WRN-28-10, a strong and stable CIFAR CNN baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class SEBlock(nn.Module):
    """Optional squeeze-and-excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class WideBasicBlock(nn.Module):
    """Pre-activation WideResNet block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout: float,
        activation: str,
        se_reduction: int,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.equal_channels = in_channels == out_channels and stride == 1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = make_activation(activation)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = make_activation(activation)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.conv2 = conv3x3(out_channels, out_channels, 1)
        self.se = SEBlock(out_channels, se_reduction) if se_reduction > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path_rate)
        self.shortcut = (
            nn.Identity()
            if self.equal_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.equal_channels:
            out = self.act1(self.bn1(x))
            shortcut = x
        else:
            out = self.act1(self.bn1(x))
            shortcut = self.shortcut(out)

        out = self.conv1(out)
        out = self.conv2(self.dropout(self.act2(self.bn2(out))))
        out = self.se(out)
        out = self.drop_path(out)
        return shortcut + out


class NetworkBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout: float,
        activation: str,
        se_reduction: int,
        drop_path_rates: list[float],
    ) -> None:
        super().__init__()
        layers = []
        for layer_idx in range(num_layers):
            layers.append(
                WideBasicBlock(
                    in_channels if layer_idx == 0 else out_channels,
                    out_channels,
                    stride if layer_idx == 0 else 1,
                    dropout,
                    activation,
                    se_reduction,
                    drop_path_rates[layer_idx],
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WideResNet(nn.Module):
    """WideResNet for 32x32 CIFAR images.

    Depth must satisfy depth = 6n + 4. WRN-28-10 uses n=4 and widen_factor=10.
    """

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        num_classes: int = 10,
        dropout: float = 0.3,
        activation: str = "relu",
        se_reduction: int = 0,
        stochastic_depth_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth should satisfy depth = 6n + 4")

        num_layers = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        total_blocks = num_layers * 3
        drop_path_rates = torch.linspace(0, stochastic_depth_rate, total_blocks).tolist()

        self.depth = depth
        self.widen_factor = widen_factor
        self.dropout = dropout
        self.activation_name = activation
        self.se_reduction = se_reduction
        self.stochastic_depth_rate = stochastic_depth_rate

        self.conv1 = conv3x3(3, widths[0])
        cursor = 0
        self.block1 = NetworkBlock(
            num_layers,
            widths[0],
            widths[1],
            stride=1,
            dropout=dropout,
            activation=activation,
            se_reduction=se_reduction,
            drop_path_rates=drop_path_rates[cursor : cursor + num_layers],
        )
        cursor += num_layers
        self.block2 = NetworkBlock(
            num_layers,
            widths[1],
            widths[2],
            stride=2,
            dropout=dropout,
            activation=activation,
            se_reduction=se_reduction,
            drop_path_rates=drop_path_rates[cursor : cursor + num_layers],
        )
        cursor += num_layers
        self.block3 = NetworkBlock(
            num_layers,
            widths[2],
            widths[3],
            stride=2,
            dropout=dropout,
            activation=activation,
            se_reduction=se_reduction,
            drop_path_rates=drop_path_rates[cursor : cursor + num_layers],
        )
        self.bn = nn.BatchNorm2d(widths[3])
        self.act = make_activation(activation)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3], num_classes)

        self._init_weights()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.act(self.bn(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.pool(x)
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


def build_model(
    depth: int = 28,
    widen_factor: int = 10,
    dropout: float = 0.3,
    activation: str = "relu",
    se_reduction: int = 0,
    stochastic_depth_rate: float = 0.0,
    num_classes: int = 10,
) -> WideResNet:
    return WideResNet(
        depth=depth,
        widen_factor=widen_factor,
        num_classes=num_classes,
        dropout=dropout,
        activation=activation,
        se_reduction=se_reduction,
        stochastic_depth_rate=stochastic_depth_rate,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def describe_model(model: nn.Module) -> dict[str, int | float | str]:
    return {
        "model": model.__class__.__name__,
        "depth": getattr(model, "depth", "unknown"),
        "widen_factor": getattr(model, "widen_factor", "unknown"),
        "dropout": getattr(model, "dropout", "unknown"),
        "activation": getattr(model, "activation_name", "unknown"),
        "se_reduction": getattr(model, "se_reduction", "unknown"),
        "stochastic_depth_rate": getattr(model, "stochastic_depth_rate", "unknown"),
        "trainable_parameters": count_parameters(model),
    }


if __name__ == "__main__":
    net = build_model()
    dummy = torch.randn(2, 3, 32, 32)
    logits = net(dummy)
    print(net)
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Trainable parameters: {count_parameters(net):,}")
