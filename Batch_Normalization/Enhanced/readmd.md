# Enhanced Batch Normalization 扩展实验

本目录用于在基础 BN 实验之上继续扩展，重点不再只是回答“BN 是否有效”，而是进一步验证：

1. BN 是否扩大可用学习率范围；
2. BN 是否让梯度变化更可预测；
3. BN 对 batch size 是否敏感；
4. BN 放置位置是否影响效果。

代码不会覆盖 `Batch_Normalization/results` 中已有结果，所有扩展实验都会保存到 `Batch_Normalization/Enhanced/results`。

## 文件说明

- `enhanced_models.py`：实现扩展模型变体。
  - `no_bn`：原始 VGG-A；
  - `bn`：标准 `Conv -> BN -> ReLU`；
  - `bn_after_relu`：`Conv -> ReLU -> BN`；
  - `bn_first_half`：只在前半部分卷积层加入 BN；
  - `bn_second_half`：只在后半部分卷积层加入 BN；
  - `groupnorm`：使用 GroupNorm 作为对照。
- `run_enhanced_experiments.py`：统一训练入口，支持学习率敏感性、batch size 敏感性、BN 位置消融和梯度预测性实验。
- `results/`：运行后自动生成，包含 CSV、模型权重和图片。

## 推荐运行方式

先进入环境和目录：

```bash
conda activate dl
cd Batch_Normalization/Enhanced
```

快速冒烟测试：

```bash
python run_enhanced_experiments.py --suite gradient --epochs 1 --gradient-lrs 1e-3 --n-train-items 512 --n-val-items 256 --batch-size 128 --no-save-model
```

## 扩展实验 1：学习率敏感性

该实验回答：BN 是否能让模型承受更大的学习率？

```bash
python run_enhanced_experiments.py --suite lr_sweep --epochs 20 --learning-rates 5e-5 1e-4 5e-4 1e-3 2e-3 3e-3 5e-3 --batch-size 128
```

输出：

- `results/summary.csv`
- `results/figures/lr_sensitivity.png`

报告中可以重点比较不同学习率下 `no_bn` 和 `bn` 的 best validation accuracy。如果高学习率下 no BN 崩溃而 BN 仍能训练，说明 BN 显著提升了学习率鲁棒性。

## 扩展实验 2：梯度预测性

该实验对应项目说明中的：

```text
Gradient predictiveness, or the change of the loss gradient.
```

脚本会记录相邻训练 step 的：

```text
grad_change_norm = ||g_t - g_{t-1}||
relative_grad_change = ||g_t - g_{t-1}|| / ||g_{t-1}||
grad_cosine = cosine(g_t, g_{t-1})
grad_diff_over_distance = ||g_t - g_{t-1}|| / ||w_t - w_{t-1}||
```

运行：

```bash
python run_enhanced_experiments.py --suite gradient --epochs 20 --gradient-lrs 1e-3 2e-3 --batch-size 128
```

输出：

- `results/summary.csv`
- `results/*/step_metrics.csv`
- `results/figures/gradient_predictiveness.png`

报告中可以说明：如果 BN 的 `relative_grad_change` 更小、`grad_cosine` 更高，则表示相邻 step 的梯度方向变化更平滑，局部线性近似更可靠。

## 扩展实验 3：Batch size 敏感性

BN 使用 mini-batch 统计量，因此 batch size 会影响均值和方差估计的稳定性。

```bash
python run_enhanced_experiments.py --suite batch_size --epochs 20 --batch-sizes 32 64 128 256 --batch-lr 1e-3
```

输出：

- `results/figures/batch_size_sensitivity.png`

报告中可以讨论：较小 batch size 下 BN 的统计估计噪声更大，表现可能波动；较大 batch size 下 BN 通常更稳定。

## 扩展实验 4：BN 位置消融

该实验比较不同归一化方式：

```bash
python run_enhanced_experiments.py --suite norm_ablation --epochs 20 --norm-lr 1e-3 --batch-size 128
```

输出：

- `results/figures/norm_ablation.png`

报告中可以比较：

- 标准 `Conv -> BN -> ReLU` 是否最优；
- `Conv -> ReLU -> BN` 是否变差；
- 只在前半部分或后半部分加 BN 是否介于 no BN 和 full BN 之间；
- GroupNorm 作为不依赖 batch 统计量的对照方法表现如何。

## 一次性全部运行

如果时间和显存允许，可以运行：

```bash
python run_enhanced_experiments.py --suite all --epochs 20
```

默认会运行较多组实验，耗时明显长于基础实验。建议先分别运行单个 suite，确认趋势后再决定是否全部跑完。

## 输出结构

运行后主要查看：

- `results/summary.csv`：所有 run 的最终汇总；
- `results/config.json`：实验配置；
- `results/<suite>/<run_id>/history.csv`：每个 epoch 的 loss/accuracy；
- `results/<suite>/<run_id>/step_metrics.csv`：每个 step 的 loss 和梯度预测性指标；
- `results/figures/*.png`：可直接放入报告的图。

## 写报告时的推荐表述

基础实验已经证明 BN 提升了 VGG-A 的表现。Enhanced 实验可以进一步说明：BN 的主要贡献不只是正则化，而是改善了优化过程。具体来说，学习率敏感性实验可以展示 BN 扩大了可用学习率范围；梯度预测性实验可以展示 BN 让相邻 step 的梯度变化更小、方向更一致；batch size 实验可以补充 BN 依赖 batch 统计量这一局限；位置消融则说明标准 `Conv -> BN -> ReLU` 是更合理的结构选择。
