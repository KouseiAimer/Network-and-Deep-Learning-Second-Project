"""PyramidNet with ShakeDrop regularization for CIFAR-10."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ShakeDropFunction(torch.autograd.Function):
    """ShakeDrop branch scaling.

    During training, each sample either keeps the residual branch or replaces
    its forward/backward scale with random coefficients. During evaluation, the
    branch is scaled by its survival probability.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, survival_prob: float, training: bool) -> torch.Tensor:
        shape = [x.size(0)] + [1] * (x.dim() - 1)
        if training:
            gate = x.new_empty(shape).bernoulli_(survival_prob)
            alpha = x.new_empty(shape).uniform_(-1.0, 1.0)
            beta = x.new_empty(shape).uniform_(0.0, 1.0)
            forward_scale = gate + (1.0 - gate) * alpha
            backward_scale = gate + (1.0 - gate) * beta
        else:
            forward_scale = x.new_full(shape, survival_prob)
            backward_scale = forward_scale

        ctx.save_for_backward(backward_scale)
        return forward_scale * x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (backward_scale,) = ctx.saved_tensors
        return backward_scale * grad_output, None, None


class ShakeDrop(nn.Module):
    def __init__(self, survival_prob: float) -> None:
        super().__init__()
        self.survival_prob = float(survival_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.survival_prob >= 1.0:
            return x
        return ShakeDropFunction.apply(x, self.survival_prob, self.training)


class PyramidShortcut(nn.Module):
    """Parameter-free shortcut with optional downsampling and zero padding."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride != 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=self.stride)

        diff = self.out_channels - self.in_channels
        if diff <= 0:
            return x

        zeros = x.new_zeros(x.size(0), diff, x.size(2), x.size(3))
        return torch.cat([x, zeros], dim=1)


class PyramidBasicBlock(nn.Module):
    """Pre-activation PyramidNet basic block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        survival_prob: float,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.shakedrop = ShakeDrop(survival_prob)
        self.shortcut = PyramidShortcut(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(self.relu1(self.bn1(x)))
        residual = self.conv2(self.relu2(self.bn2(residual)))
        residual = self.shakedrop(residual)
        return self.shortcut(x) + residual


class PyramidNetCIFAR(nn.Module):
    """CIFAR PyramidNet with basic residual blocks.

    For the basic block variant, depth satisfies depth = 6n + 2.
    Channel width grows gradually from 16 to 16 + alpha.
    """

    def __init__(
        self,
        depth: int = 110,
        alpha: int = 270,
        num_classes: int = 10,
        final_survival_prob: float = 0.5,
    ) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("PyramidNet basic depth should satisfy depth = 6n + 2")
        if not 0 < final_survival_prob <= 1:
            raise ValueError("final_survival_prob must be in (0, 1]")

        self.depth = depth
        self.alpha = alpha
        self.final_survival_prob = final_survival_prob
        self.blocks_per_stage = (depth - 2) // 6
        self.total_blocks = self.blocks_per_stage * 3
        self.add_rate = alpha / self.total_blocks

        self.in_channels = 16
        self.featuremap_dim = 16.0
        self.block_index = 0

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(self.blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(self.blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(self.blocks_per_stage, stride=2)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

        self._init_weights()

    def _survival_prob(self) -> float:
        self.block_index += 1
        drop_fraction = self.block_index / self.total_blocks
        return 1.0 - drop_fraction * (1.0 - self.final_survival_prob)

    def _make_layer(self, num_blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for block_id in range(num_blocks):
            block_stride = stride if block_id == 0 else 1
            self.featuremap_dim += self.add_rate
            out_channels = int(round(self.featuremap_dim))
            layers.append(
                PyramidBasicBlock(
                    self.in_channels,
                    out_channels,
                    block_stride,
                    self._survival_prob(),
                )
            )
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
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
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)


def pyramidnet_shakedrop(
    depth: int = 110,
    alpha: int = 270,
    num_classes: int = 10,
    final_survival_prob: float = 0.5,
) -> PyramidNetCIFAR:
    return PyramidNetCIFAR(
        depth=depth,
        alpha=alpha,
        num_classes=num_classes,
        final_survival_prob=final_survival_prob,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def describe_model(model: nn.Module) -> dict[str, int | float | str]:
    return {
        "model": model.__class__.__name__,
        "depth": getattr(model, "depth", "unknown"),
        "alpha": getattr(model, "alpha", "unknown"),
        "blocks_per_stage": getattr(model, "blocks_per_stage", "unknown"),
        "total_blocks": getattr(model, "total_blocks", "unknown"),
        "final_survival_prob": getattr(model, "final_survival_prob", "unknown"),
        "trainable_parameters": count_parameters(model),
    }


if __name__ == "__main__":
    net = pyramidnet_shakedrop()
    dummy = torch.randn(2, 3, 32, 32)
    logits = net(dummy)
    print(net)
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Trainable parameters: {count_parameters(net):,}")
