"""Improved DenseNet variants for the final CIFAR-10 network."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def make_activation(name: str, inplace: bool = True) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    if name == "elu":
        return nn.ELU(inplace=inplace)
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish(inplace=inplace)
    raise ValueError(f"Unsupported activation: {name}")


class DropPath(nn.Module):
    """Per-sample stochastic depth for residual/new-feature branches."""

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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

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


class DenseLayer(nn.Module):
    """DenseNet-BC layer with optional SE, dropout, and stochastic depth."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        drop_path_rate: float,
        activation: str,
        se_reduction: int,
    ) -> None:
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            make_activation(activation),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            make_activation(activation),
            nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.se = SEBlock(growth_rate, reduction=se_reduction) if se_reduction > 0 else nn.Identity()
        self.drop_rate = float(drop_rate)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.net(x)
        new_features = self.se(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        new_features = self.drop_path(new_features)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        drop_path_rates: list[float],
        activation: str,
        se_reduction: int,
    ) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for layer_idx in range(num_layers):
            layers.append(
                DenseLayer(
                    channels,
                    growth_rate,
                    bn_size,
                    drop_rate,
                    drop_path_rates[layer_idx],
                    activation,
                    se_reduction,
                )
            )
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Transition(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        transition_dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.BatchNorm2d(in_channels),
            make_activation(activation),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        ]
        if transition_dropout > 0:
            layers.append(nn.Dropout2d(p=transition_dropout))
        layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MyDenseNet(nn.Module):
    """SE-DenseNet-BC for CIFAR-10.

    This model keeps DenseNet-BC's feature reuse, then adds SE attention,
    stochastic depth, optional classifier dropout, and configurable activation
    functions for final-project ablations.
    """

    def __init__(
        self,
        depth: int = 190,
        growth_rate: int = 40,
        compression: float = 0.5,
        num_classes: int = 10,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        transition_dropout: float = 0.0,
        stochastic_depth_rate: float = 0.2,
        activation: str = "silu",
        se_reduction: int = 16,
        classifier_hidden: int = 512,
        classifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not 0 < compression <= 1:
            raise ValueError("compression must be in (0, 1]")
        if (depth - 4) % 6 != 0:
            raise ValueError("DenseNet-BC depth should satisfy depth = 6n + 4")

        layers_per_block = (depth - 4) // 6
        total_layers = layers_per_block * 3
        drop_path_rates = torch.linspace(0, stochastic_depth_rate, total_layers).tolist()

        self.depth = depth
        self.growth_rate = growth_rate
        self.compression = compression
        self.layers_per_block = layers_per_block
        self.activation_name = activation
        self.se_reduction = se_reduction
        self.stochastic_depth_rate = stochastic_depth_rate

        channels = 2 * growth_rate
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            make_activation(activation),
        )

        cursor = 0
        self.block1 = DenseBlock(
            layers_per_block,
            channels,
            growth_rate,
            bn_size,
            drop_rate,
            drop_path_rates[cursor : cursor + layers_per_block],
            activation,
            se_reduction,
        )
        cursor += layers_per_block
        channels = self.block1.out_channels
        out_channels = math.floor(channels * compression)
        self.trans1 = Transition(channels, out_channels, activation, transition_dropout)
        channels = out_channels

        self.block2 = DenseBlock(
            layers_per_block,
            channels,
            growth_rate,
            bn_size,
            drop_rate,
            drop_path_rates[cursor : cursor + layers_per_block],
            activation,
            se_reduction,
        )
        cursor += layers_per_block
        channels = self.block2.out_channels
        out_channels = math.floor(channels * compression)
        self.trans2 = Transition(channels, out_channels, activation, transition_dropout)
        channels = out_channels

        self.block3 = DenseBlock(
            layers_per_block,
            channels,
            growth_rate,
            bn_size,
            drop_rate,
            drop_path_rates[cursor : cursor + layers_per_block],
            activation,
            se_reduction,
        )
        channels = self.block3.out_channels

        self.norm = nn.BatchNorm2d(channels)
        self.act = make_activation(activation)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if classifier_hidden > 0:
            self.classifier = nn.Sequential(
                nn.Linear(channels, classifier_hidden),
                make_activation(activation, inplace=False),
                nn.Dropout(p=classifier_dropout),
                nn.Linear(classifier_hidden, num_classes),
            )
        else:
            self.classifier = nn.Linear(channels, num_classes)

        self._init_weights()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.block3(x)
        return self.act(self.norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

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


def build_model(
    depth: int = 190,
    growth_rate: int = 40,
    compression: float = 0.5,
    activation: str = "silu",
    se_reduction: int = 16,
    stochastic_depth_rate: float = 0.2,
    classifier_hidden: int = 512,
    classifier_dropout: float = 0.2,
    drop_rate: float = 0.0,
    transition_dropout: float = 0.0,
    num_classes: int = 10,
) -> MyDenseNet:
    return MyDenseNet(
        depth=depth,
        growth_rate=growth_rate,
        compression=compression,
        activation=activation,
        se_reduction=se_reduction,
        stochastic_depth_rate=stochastic_depth_rate,
        classifier_hidden=classifier_hidden,
        classifier_dropout=classifier_dropout,
        drop_rate=drop_rate,
        transition_dropout=transition_dropout,
        num_classes=num_classes,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def describe_model(model: nn.Module) -> dict[str, int | float | str]:
    return {
        "model": model.__class__.__name__,
        "depth": getattr(model, "depth", "unknown"),
        "growth_rate": getattr(model, "growth_rate", "unknown"),
        "compression": getattr(model, "compression", "unknown"),
        "layers_per_block": getattr(model, "layers_per_block", "unknown"),
        "activation": getattr(model, "activation_name", "unknown"),
        "se_reduction": getattr(model, "se_reduction", "unknown"),
        "stochastic_depth_rate": getattr(model, "stochastic_depth_rate", "unknown"),
        "trainable_parameters": count_parameters(model),
    }


if __name__ == "__main__":
    net = build_model(depth=40, growth_rate=16)
    dummy = torch.randn(2, 3, 32, 32)
    logits = net(dummy)
    print(net)
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Trainable parameters: {count_parameters(net):,}")
