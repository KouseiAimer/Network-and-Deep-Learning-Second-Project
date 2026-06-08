# Batch Normalization 实验说明

本文件夹是对 `codes/VGG_BatchNorm` 中示例脚手架的完整化实现，目标是完成项目第二部分 **Batch Normalization (30%)** 的实验代码。原始代码给出了 VGG-A、CIFAR-10 加载器和 `VGG_Loss_Landscape.py` 的框架，但仍有若干空缺，例如 `VGG_A_BatchNorm` 未实现、训练循环没有记录 loss/accuracy、loss landscape 的 `min_curve/max_curve` 需要自己补充。本目录下的代码将这些部分整理成一套可复现的实验流程。

## 文件结构

- `models.py`：实现 CIFAR-10 版本的 VGG-A，以及在每个卷积层后加入 `BatchNorm2d` 的 `VGG_A_BatchNorm`。
- `data_utils.py`：封装 CIFAR-10 数据加载，修正原示例中 `PartialDataset.__getitem__` 的问题，并支持小数据子集调试。
- `run_bn_experiments.py`：完整训练与可视化脚本。它会分别训练无 BN 和有 BN 的 VGG-A，并在多组学习率下记录 batch loss、epoch accuracy、梯度范数，最后生成 loss landscape 包络图。
- `results/`：运行脚本后自动生成，保存 CSV、模型权重和图片。

## 已完成内容

1. 实现 `VGG_A`

   结构与作业给出的 VGG-A 思路一致：输入为 `32x32x3` 的 CIFAR-10 图像，卷积部分使用 VGG-A 配置，最后经过三个全连接层输出 10 类分类结果。

2. 实现 `VGG_A_BatchNorm`

   在每个卷积层后加入：

   ```text
   Conv2d -> BatchNorm2d -> ReLU
   ```

   由于 BN 层本身带有可学习的仿射参数，带 BN 的卷积层默认不使用 bias。

3. 完整训练循环

   `run_bn_experiments.py` 会记录：

   - 每个 batch 的训练 loss；
   - 每个 epoch 的 train loss、train accuracy、validation loss、validation accuracy；
   - 每个 batch 的全局梯度范数；
   - 每次运行的 best/last 模型权重。

4. Loss landscape 包络图

   对同一个模型使用多组学习率训练，例如：

   ```text
   1e-3, 2e-3, 5e-4, 1e-4
   ```

   然后在同一个训练 step 上统计所有学习率对应 loss 的最小值与最大值，得到：

   ```text
   min_curve, max_curve
   ```

   最终用 `matplotlib.pyplot.fill_between()` 画出无 BN 和有 BN 的 loss 波动区域，方便观察 BN 是否让训练过程更稳定。

5. 可用于报告的输出

   运行后会生成：

   - `results/summary.csv`：每次实验的最终指标汇总；
   - `results/*/history.csv`：每个 epoch 的训练/验证指标；
   - `results/*/batch_losses.csv`：每个 batch 的 loss；
   - `results/landscape_no_bn.csv`、`results/landscape_bn.csv`：loss landscape 包络数据；
   - `results/figures/training_curves_first_lr.png`：首个学习率下的训练曲线对比；
   - `results/figures/loss_landscape_envelope.png`：BN 与 no-BN 的 loss landscape 对比；
   - `results/figures/gradient_norms_first_lr.png`：首个学习率下的梯度范数对比。

## 运行方式

本机建议先进入 `dl` 环境：

```bash
conda activate dl
cd "Batch Normalization"
```

建议先做快速冒烟测试，确认环境、数据集和脚本都能正常运行：

```bash
python run_bn_experiments.py --epochs 1 --learning-rates 1e-3 --n-train-items 512 --n-val-items 256 --batch-size 128 --num-workers 0 --no-save-model
```

正式实验可以使用：

```bash
python run_bn_experiments.py --epochs 20 --learning-rates 1e-3 2e-3 5e-4 1e-4 --batch-size 128 --num-workers 0
```

如果有 CUDA，可以显式指定：

```bash
python run_bn_experiments.py --device cuda --epochs 20 --learning-rates 1e-3 2e-3 5e-4 1e-4
```

如果想更快得到趋势，可以使用部分训练集：

```bash
python run_bn_experiments.py --epochs 10 --learning-rates 1e-3 2e-3 5e-4 1e-4 --n-train-items 5000 --n-val-items 1000
```

## 写报告时可以怎么描述

第二部分报告建议按下面逻辑展开：

1. 简要说明 BN 的计算方式：对每个通道统计 mini-batch 和空间位置上的均值、方差，归一化后再做可学习仿射变换。
2. 说明实验设置：CIFAR-10、VGG-A、是否使用 BN、优化器、学习率列表、epoch 数、batch size。
3. 展示普通训练曲线：比较 BN 和 no-BN 的训练 loss、验证 accuracy、收敛速度。
4. 展示 loss landscape 包络图：比较两种模型在多学习率下 loss 波动区域的宽窄。
5. 解释实验现象：如果 BN 的包络更窄、训练更平稳，可以说明 BN 让优化景观更平滑，对学习率变化更鲁棒；这与 Santurkar et al. 2018 中“BN improves optimization by smoothing the loss landscape”的观点一致。

## 注意事项

- 为保证对比公平，脚本对 BN 和 no-BN 使用相同的数据、优化器、学习率列表和随机种子。
- `--augment` 默认关闭，因为本实验重点是比较 BN 对优化过程的影响；如果只想追求更高准确率，可以打开数据增强，但报告里要说明。
- 完整运行需要训练 2 个模型乘以学习率数量。默认 4 个学习率时，一共会训练 8 次。
- 若机器较慢，建议先用 `--n-train-items` 和 `--n-val-items` 观察趋势，再决定是否跑完整 CIFAR-10。
