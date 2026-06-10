# 神经网络与深度学习 Project 2

本仓库对应复旦大学《神经网络与深度学习》Project 2。项目分为两部分：

- 第一部分：在 CIFAR-10 上训练并优化图像分类网络，报告最佳测试错误率、网络结构、消融实验和可视化分析。
- 第二部分：完成 Batch Normalization 实验，比较 VGG-A 有/无 BN 的训练行为，并解释 BN 如何帮助优化。

GitHub 代码仓库：

```text
https://github.com/KouseiAimer/Network-and-Deep-Learning-Second-Project
```

模型权重仓库：

```text
https://modelscope.cn/models/KouseiAimer/Network-and-Deep-Learning-Second-Project
```

权重文件较大，不放入 GitHub；ModelScope 中的权重路径与本 GitHub 仓库的相对路径保持一致。例如 ModelScope 中的
`Ablation/results/final/best_from_ablation/weights/best.pt` 对应本仓库中的
`Ablation/ablation.py`、`Ablation/final.py` 和同目录下的 `config.json`、`summary.json`。

## 0. 最终模型快速验证

根目录下的 `quick_check.py` 用于快速复核第一部分最终模型。它会自动读取最终 Ablation 实验的配置和权重：

```text
Ablation/results/final/best_from_ablation/config.json
Ablation/results/final/best_from_ablation/weights/best.pt
```

默认优先加载 checkpoint 中的 `ema_state`，然后在 CIFAR-10 test set 上计算一次完整错误率。

运行方式：

```bash
conda activate dl
python quick_check.py --num-workers 0
```

如果本地还没有 CIFAR-10 数据集，可以加上：

```bash
python quick_check.py --num-workers 0 --download
```

已经验证得到的输出为：

```text
test_accuracy : 97.15%
test_error    : 2.85%
correct/total : 9715/10000
```

这与 `Ablation/results/final/best_from_ablation/summary.json` 中记录的最终结果一致。若只想检查模型加载流程，可以使用：

```bash
python quick_check.py --max-batches 1 --num-workers 0
```

## 1. 项目要求与完成情况

项目说明文件为 `project_2_2026.pdf`。其核心要求包括：

| 项目要求 | 本仓库完成位置 |
|---|---|
| CIFAR-10 分类，报告最佳测试错误率 | `Ablation/`，最终错误率 `2.85%` |
| 网络包含全连接层、二维卷积层、二维池化层、激活函数 | `Ablation/ablation.py` 中的 SE-DenseNet-BC 满足全部要求 |
| 至少包含 BN、Dropout、残差连接或其他组件之一 | 最终模型包含 BatchNorm、Dropout、Dense connection、SE attention、stochastic depth |
| 尝试不同滤波器数量 | `Ablation/results/ablation/capacity/`，对比 `growth_rate=40/32/24` |
| 尝试不同损失函数 | `Ablation/results/ablation/loss/`，对比 CE 与 Focal Loss |
| 尝试不同激活函数 | `Ablation/results/ablation/activation/`，对比 SiLU、ReLU、GELU |
| 尝试不同优化器 | `Ablation/results/ablation/optimizer/`，对比 SGD、AdamW、RMSprop |
| 网络洞察与可视化 | 训练曲线、混淆矩阵、逐类准确率、误分类样例、置信度分布、卷积核、Grad-CAM、loss landscape |
| VGG-A 有/无 BN 对比 | `Batch_Normalization/` |
| 解释 BN 如何帮助优化 | `Batch_Normalization/Enhanced/` 与 `Batch_Normalization/Visual/` |

## 2. 仓库结构

```text
.
├── README.md
├── quick_check.py
├── project_2_2026.pdf
├── ModelScope_README.md
├── Ablation/
│   ├── ablation.py
│   ├── final.py
│   ├── visual_ablation.py
│   ├── visual_final.py
│   ├── readmd.md
│   └── results/
├── Batch_Normalization/
│   ├── models.py
│   ├── data_utils.py
│   ├── run_bn_experiments.py
│   ├── readmd.md
│   ├── report/
│   ├── results/
│   ├── Enhanced/
│   └── Visual/
├── classic_CNN/
├── classic_ResNet-110/
├── classic_DenseNet/
├── classic_WideResNet/
├── classic_PyramidNet+ShakeDrop/
├── my_DenseNet/
├── my_final_CNN/
└── codes/VGG_BatchNorm/
```

主要目录说明：

- `Ablation/`：第一部分最终模型、50 epoch 消融实验、300 epoch 最终复跑和可视化脚本。最终采用该目录下的模型。
- `Batch_Normalization/`：第二部分 BN 实验。基础实验比较 VGG-A 与 VGG-A-BN；`Enhanced/` 做学习率敏感性和梯度预测性分析；`Visual/` 生成报告级可视化。
- `classic_*`：若干经典网络 baseline，包括 classic CNN、ResNet-110、DenseNet-BC、WideResNet、PyramidNet + ShakeDrop。
- `my_DenseNet/`：早期最终模型候选，SE-DenseNet-BC-190-40，后续由 `Ablation/` 中更轻的配置替代。
- `my_final_CNN/`：另一个强模型尝试，保留训练与失败分析材料。
- `codes/VGG_BatchNorm/`：课程提供的 VGG/BatchNorm 示例脚手架。
- `ModelScope_README.md`：ModelScope 权重仓库的中文说明，记录权重路径与 GitHub 代码的对应关系。

## 3. 数据集与环境

数据集使用 CIFAR-10。训练集 50,000 张，测试集 10,000 张，共 10 类：

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

数据集默认通过 `torchvision.datasets.CIFAR10` 下载到本地 `data/` 目录。数据集不上传 GitHub。

CIFAR-10 官方数据集链接：

```text
https://www.cs.toronto.edu/~kriz/cifar.html
```

建议环境：

```bash
conda activate dl
```

主要依赖：

```text
python
torch
torchvision
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
```

如果在 Windows 上运行 DataLoader 遇到多进程问题，可以统一加 `--num-workers 0`。

## 4. 第一部分：CIFAR-10 最终模型

### 4.1 最终模型结构

第一部分最终采用 `Ablation/` 中重新实现的模型：

```text
SE-DenseNet-BC-190-24
depth = 190
growth_rate = 24
compression = 0.5
activation = GELU
SE reduction = 16
stochastic depth rate = 0.2
classifier hidden dim = 512
classifier dropout = 0.2
loss = CrossEntropy + label smoothing 0.1
optimizer = SGD + momentum
scheduler = cosine
augmentation = AutoAugment + Cutout + CutMix
EMA decay = 0.999
trainable parameters = 9,978,062
```

选择该模型的原因是：原先的 `SE-DenseNet-BC-190-40` 虽然强，但参数量约 26.76M；消融实验显示 `growth_rate=24` 可以显著降低参数量，并保持很强的泛化能力。最终 300 epoch 复跑后，`growth_rate=24 + GELU` 的组合达到最高测试准确率，同时模型规模更适合报告中关于参数量与性能的比较。

### 4.2 最终结果

最终实验目录：

```text
Ablation/results/final/best_from_ablation/
```

最终权重在 ModelScope：

```text
Ablation/results/final/best_from_ablation/weights/best.pt
Ablation/results/final/best_from_ablation/weights/last.pt
```

结果摘要：

| 指标 | 数值 |
|---|---:|
| Best epoch | 284 |
| Best test accuracy | 97.15% |
| Best test error | 2.85% |
| Best test loss | 0.5846 |
| Final epoch | 300 |
| Final test accuracy | 97.13% |
| Final test loss | 0.5838 |
| 参数量 | 9,978,062 |

### 4.3 Baseline 与模型探索结果

| 模型 | 目录 | Best test accuracy | Test error |
|---|---|---:|---:|
| Classic CNN | `classic_CNN/` | 74.04% | 25.96% |
| ResNet-110 | `classic_ResNet-110/` | 93.40% | 6.60% |
| DenseNet-BC-100-24 | `classic_DenseNet/` | 96.42% | 3.58% |
| WideResNet-28-10 | `classic_WideResNet/` | 96.63% | 3.37% |
| SE-DenseNet-BC-190-40 | `my_DenseNet/` | 96.68% | 3.32% |
| SE-StochasticDepth-WRN-40-10 | `my_final_CNN/` | 91.08% | 8.92% |
| Final SE-DenseNet-BC-190-24 | `Ablation/` | 97.15% | 2.85% |

其中 `my_final_CNN/` 保留为一次结构尝试和失败分析：该模型使用较强正则与增强，但最终明显低于 DenseNet 系列，因此没有作为最终提交模型。

### 4.4 消融实验

消融实验统一使用 50 epoch 短预算，目的是比较设计选择对早期收敛和泛化趋势的影响，而不是替代最终 300 epoch 训练。

消融结果总表：

```text
Ablation/results/ablation/summary.csv
Ablation/results/ablation/best_config.json
```

核心消融结果：

| 组别 | 实验 | Best test accuracy | 参数量 |
|---|---|---:|---:|
| baseline | `SE-DenseNet-BC-190-40 + SiLU + CE + SGD` | 91.13% | 26,763,454 |
| capacity | `growth_rate=32` | 91.63% | 17,357,022 |
| capacity | `growth_rate=24` | 91.64% | 9,978,062 |
| activation | `ReLU` | 92.99% | 26,763,454 |
| activation | `GELU` | 93.13% | 26,763,454 |
| loss | `Focal Loss` | 89.35% | 26,763,454 |
| optimizer | `AdamW` | 90.53% | 26,763,454 |
| optimizer | `RMSprop` | 10.00% | 26,763,454 |

消融结论：

- `GELU` 在短训练预算下优于默认 `SiLU`，最终模型采用 GELU。
- `growth_rate=24` 在大幅减少参数量的同时保持了很好的早期泛化表现，最终模型采用该容量。
- Focal Loss 不适合本实验的最终配方，仍使用 CrossEntropy + label smoothing。
- SGD + momentum 比 AdamW 和 RMSprop 更稳，最终继续使用 SGD。
- RMSprop 在该大模型和学习率设置下训练失败，说明优化器选择需要与学习率、正则化和模型规模配套。

### 4.5 第一部分可视化

最终可视化输出位于：

```text
Ablation/results/final/best_from_ablation/visualizations/
Ablation/results/final/best_from_ablation/loss_landscape/
```

包括：

- `curves.png`：训练/测试 loss 与 accuracy 曲线；
- `confusion_matrix.png`：混淆矩阵；
- `per_class_accuracy.png`：逐类准确率；
- `confidence_histogram.png`：预测置信度分布；
- `misclassified_examples.png`：误分类样例；
- `first_conv_filters.png`：第一层卷积核；
- `gradcam_examples.png`：Grad-CAM 可解释性图；
- `loss_landscape.png`：最终模型局部 loss landscape。

这些图对应项目要求中的“展示对网络的洞察，例如滤波器可视化、损失景观、网络解释等”。

## 5. 第一部分复现实验命令

从项目根目录运行。

消融实验：

```bash
conda activate dl
python Ablation/ablation.py --epochs 50 --amp
python Ablation/visual_ablation.py
```

最终 300 epoch 复跑：

```bash
python Ablation/final.py --epochs 300 --amp
python Ablation/visual_final.py
```

快速冒烟测试：

```bash
python Ablation/ablation.py --epochs 1 --subset 256 --eval-max-batches 2 --num-workers 0 --rerun
python Ablation/visual_ablation.py
python Ablation/final.py --epochs 1 --num-workers 0 --amp
python Ablation/visual_final.py --max-batches 2 --landscape-max-batches 1 --landscape-points 5
```

只验证最终权重：

```bash
python quick_check.py --num-workers 0
```

## 6. 第二部分：Batch Normalization

第二部分位于：

```text
Batch_Normalization/
```

完整中文报告：

```text
Batch_Normalization/report/report.pdf
Batch_Normalization/report/report.tex
```

### 6.1 实验目标

根据项目要求，第二部分需要完成两件事：

1. 比较 VGG-A 与 VGG-A-BatchNorm 在 CIFAR-10 上的性能和训练特征。
2. 解释 BN 如何帮助优化，重点观察 loss landscape、gradient predictiveness 和梯度随距离变化的最大差异。

本仓库实现了三层实验：

- `Batch_Normalization/`：基础 VGG-A 有/无 BN 对比。
- `Batch_Normalization/Enhanced/`：学习率敏感性、梯度预测性等扩展实验。
- `Batch_Normalization/Visual/`：论文风格曲线、3D loss surface、激活分布 ridgeline 等可视化。

### 6.2 VGG-A 与 VGG-A-BN

基础模型：

```text
VGG-A:
Conv2d -> ReLU -> MaxPool blocks -> FC classifier

VGG-A-BN:
Conv2d -> BatchNorm2d -> ReLU -> MaxPool blocks -> FC classifier
```

参数量：

| 模型 | 参数量 |
|---|---:|
| VGG-A | 9,750,922 |
| VGG-A-BN | 9,753,674 |

两者参数量几乎相同，因此性能差异主要来自 BN 对优化过程的影响，而不是模型容量增加。

基础实验配置：

| 项目 | 设置 |
|---|---|
| 数据集 | CIFAR-10 |
| Epoch | 20 |
| Batch size | 128 |
| 优化器 | Adam |
| Weight decay | 5e-4 |
| 学习率 | 1e-3, 2e-3, 5e-4, 1e-4 |
| 数据增强 | 无 |

基础结果：

| 学习率 | 模型 | Best val accuracy | Best epoch | Final val accuracy |
|---:|---|---:|---:|---:|
| 1e-4 | VGG-A | 76.84% | 13 | 76.28% |
| 1e-4 | VGG-A-BN | 74.71% | 17 | 74.34% |
| 5e-4 | VGG-A | 79.25% | 19 | 78.12% |
| 5e-4 | VGG-A-BN | 81.82% | 19 | 78.17% |
| 1e-3 | VGG-A | 77.97% | 18 | 77.79% |
| 1e-3 | VGG-A-BN | 82.18% | 12 | 81.03% |
| 2e-3 | VGG-A | 10.00% | 1 | 10.00% |
| 2e-3 | VGG-A-BN | 81.36% | 19 | 79.53% |

主要结论：

- 在 `lr=1e-3` 时，BN 将最佳验证准确率从 77.97% 提升到 82.18%。
- 在 `lr=2e-3` 时，无 BN 模型退化到随机猜测的 10.00%，而 BN 模型仍达到 81.36%。
- BN 的优势主要体现在中等和较大学习率下：它扩大了可用学习率范围，使训练对超参数更鲁棒。

### 6.3 Loss landscape 与优化稳定性

项目要求建议使用多个学习率训练模型，然后在同一训练 step 上统计所有学习率 run 的 loss 最大值和最小值：

```text
max_curve(t) = max_i loss_i(t)
min_curve(t) = min_i loss_i(t)
```

两条曲线之间的面积可理解为训练 loss 对学习率变化的波动包络。包络越窄，说明训练过程越稳定。

本仓库中的结果：

| 模型 | 平均包络宽度 | 中位数宽度 | 90 分位宽度 |
|---|---:|---:|---:|
| VGG-A | 1.900 | 2.077 | 2.262 |
| VGG-A-BN | 0.375 | 0.355 | 0.564 |

BN 将平均 loss 包络宽度从 1.900 降低到 0.375，约减少 80.3%。这说明 BN 不只是提升最终精度，更重要的是使不同学习率下的训练轨迹更一致，优化过程更稳定。

### 6.4 学习率敏感性

Enhanced 实验进一步扩展学习率列表：

```text
5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3
```

关键结果：

| 学习率 | VGG-A best acc | VGG-A-BN best acc | BN 提升 |
|---:|---:|---:|---:|
| 5e-5 | 75.09% | 70.58% | -4.51 pp |
| 1e-4 | 76.84% | 74.71% | -2.13 pp |
| 5e-4 | 79.25% | 81.82% | +2.57 pp |
| 1e-3 | 77.97% | 82.18% | +4.21 pp |
| 2e-3 | 10.00% | 81.36% | +71.36 pp |
| 3e-3 | 10.00% | 79.46% | +69.46 pp |
| 5e-3 | 65.49% | 75.00% | +9.51 pp |

解释：

- 小学习率下，无 BN 已经可以稳定训练，BN 不一定带来优势。
- 当学习率变大时，无 BN 更容易崩溃；BN 能维持有效学习。
- 因此，BN 的核心价值不是“任何情况下都提高准确率”，而是改善优化条件，使模型可以使用更激进的学习率。

### 6.5 梯度预测性与激活可视化

`Batch_Normalization/Enhanced/` 记录了相邻 step 的梯度变化，包括：

```text
grad_change_norm
relative_grad_change
grad_cosine
grad_diff_over_distance
```

这些指标用于观察局部线性近似是否稳定。实验结论是：梯度指标必须和准确率一起解释。在较大学习率下，无 BN 模型可能因为训练失败而出现很小的梯度范数，这并不表示优化稳定；真正重要的是 BN 能在较大学习率下保持有效训练和较连续的梯度方向。

`Batch_Normalization/Visual/` 还提供：

- `paper_style_accuracy_panels.png`：论文风格训练/测试准确率曲线；
- `loss_surface_3d_no_bn_vs_bn.png`：局部二维参数切片的 3D loss surface；
- `activation_distribution_ridgeline.png`：中间激活分布；
- `activation_distribution_stats.csv`：激活统计表。

### 6.6 第二部分复现实验命令

基础 BN 实验：

```bash
conda activate dl
python Batch_Normalization/run_bn_experiments.py --epochs 20 --learning-rates 1e-3 2e-3 5e-4 1e-4 --batch-size 128 --num-workers 0
```

快速冒烟测试：

```bash
python Batch_Normalization/run_bn_experiments.py --epochs 1 --learning-rates 1e-3 --n-train-items 512 --n-val-items 256 --batch-size 128 --num-workers 0 --no-save-model
```

学习率敏感性：

```bash
python Batch_Normalization/Enhanced/run_enhanced_experiments.py --suite lr_sweep --epochs 20 --learning-rates 5e-5 1e-4 5e-4 1e-3 2e-3 3e-3 5e-3 --batch-size 128 --num-workers 0
```

梯度预测性：

```bash
python Batch_Normalization/Enhanced/run_enhanced_experiments.py --suite gradient --epochs 20 --gradient-lrs 1e-3 2e-3 --batch-size 128 --num-workers 0
```

生成基础实验可视化：

```bash
python Batch_Normalization/Visual/visualize_bn.py --mode all --accuracy-lrs 1e-3 2e-3 --surface-grid 17 --surface-samples 512 --activation-samples 1024
```

如果只想生成最稳妥、耗时较少的报告图：

```bash
python Batch_Normalization/Visual/visualize_bn.py --mode accuracy
python Batch_Normalization/Visual/visualize_bn.py --mode activation --activation-samples 1024
```

## 7. 权重下载与路径说明

所有 `.pt` 权重已上传到 ModelScope：

```text
https://modelscope.cn/models/KouseiAimer/Network-and-Deep-Learning-Second-Project
```

重要权重路径：

| 权重路径 | 对应代码 | 说明 |
|---|---|---|
| `Ablation/results/final/best_from_ablation/weights/` | `Ablation/ablation.py`, `Ablation/final.py` | 最终模型，97.15% accuracy |
| `Ablation/results/ablation/*/*/weights/` | `Ablation/ablation.py` | 50 epoch 消融实验 |
| `classic_ResNet-110/weights/resnet110/` | `classic_ResNet-110/model.py` | ResNet-110 baseline |
| `classic_DenseNet/weights/densenet_bc_100_24/` | `classic_DenseNet/model.py` | DenseNet-BC baseline |
| `classic_WideResNet/final_result/wrn28_10/weights/` | `classic_WideResNet/model.py` | WRN-28-10 baseline |
| `my_DenseNet/final_result/final_se_densenet/weights/` | `my_DenseNet/model.py` | 早期 SE-DenseNet-BC-190-40 |
| `Batch_Normalization/results/*/` | `Batch_Normalization/models.py` | VGG-A 有/无 BN 基础实验 |
| `Batch_Normalization/Enhanced/results/*/*/` | `Batch_Normalization/Enhanced/enhanced_models.py` | BN 扩展实验 |

下载后请保持与 GitHub 中相同的相对路径，这样脚本可以直接读取配置和 checkpoint。

## 8. 报告与结果文件

第一部分主要报告材料：

```text
Ablation/results/ablation/figures/ablation_summary.md
Ablation/results/ablation/figures/ablation_best_accuracy.png
Ablation/results/ablation/figures/ablation_learning_curves.png
Ablation/results/final/best_from_ablation/summary.json
Ablation/results/final/best_from_ablation/curves.png
Ablation/results/final/best_from_ablation/visualizations/
Ablation/results/final/best_from_ablation/loss_landscape/
```

第二部分主要报告材料：

```text
Batch_Normalization/report/report.pdf
Batch_Normalization/results/summary.csv
Batch_Normalization/results/figures/
Batch_Normalization/Enhanced/results/summary.csv
Batch_Normalization/Enhanced/results/figures/
Batch_Normalization/Visual/figures/
```

根目录的 `ModelScope_README.md` 是上传到 ModelScope 的中文权重说明，可用于确认每个权重文件对应的模型代码。

## 9. GitHub 提交说明

GitHub 保存：

- 源代码；
- 配置文件；
- `summary.json`、`history.csv`、`summary.csv` 等轻量结果；
- 训练曲线、可视化图片；
- LaTeX 报告源码和 PDF。

GitHub 不保存：

- CIFAR-10 数据集；
- `.pt` 模型权重；
- 大体积缓存文件。

大文件权重统一通过 ModelScope 提供。最终报告中同时给出 GitHub 代码链接、ModelScope 权重链接和 CIFAR-10 数据集链接，以满足项目说明中的提交要求。
