# 最终 CNN：SE-StochasticDepth WideResNet

本目录实现 CIFAR-10 第一部分任务的最终 CNN 网络。模型以此前表现最稳定的 WideResNet 为主干，在此基础上加入 SE 通道注意力、stochastic depth、Dropout、EMA、RandAugment、Cutout、CutMix、warmup cosine 学习率和可选 TTA，目标是在 20GB 显存服务器上冲击约 97% 的测试准确率。

默认最终模型：

```text
FinalCNN = SE-SD-WRN-40-10
depth = 40
widen_factor = 10
activation = SiLU
SE reduction = 16
stochastic depth rate = 0.1
dropout = 0.3
epochs = 300
trainable parameters = 56,252,094
```

该模型包含项目要求的所有基础组件：全连接层、二维卷积层、二维池化层和激活函数；同时包含 BatchNorm、Dropout、残差连接、SE 注意力和 stochastic depth。

## 文件说明

- `model.py`：最终 CNN 结构，包含 SE、DropPath、pre-activation WideResNet block。
- `train.py`：训练、验证、checkpoint、EMA、CutMix/Mixup、TTA、干净训练集评估、优化器和 loss 选择。
- `plot_results.py`：根据 `history.csv` 绘制 loss、accuracy、error、学习率和泛化 gap 曲线，并生成 `summary.json`。
- `visualize.py`：生成混淆矩阵、逐类准确率、置信度分布、误分类样例、第一层卷积滤波器和 Grad-CAM。
- `loss_landscape.py`：在最佳 checkpoint 附近绘制一维 loss landscape。
- `ablation_plan.md`：用于满足项目第 1 部分第 4、5 点的消融实验命令。
- `final_result/`：服务器训练输出目录。

## 推荐训练命令

在项目根目录运行：

```bash
conda activate dl
python my_final_CNN/train.py --amp --eval-tta
```

默认输出路径：

```text
my_final_CNN/final_result/final_swrn40_10/
```

训练完成后，目录中会包含：

```text
config.json
history.csv
curves.png
summary.json
weights/best.pt
weights/last.pt
```

其中 `.pt` 权重文件较大，不应上传到 GitHub，后续请上传到网盘，并在最终报告中附链接。

## 训练日志如何理解

默认启用了 CutMix、RandAugment、Cutout 和 label smoothing，因此日志中的 `train acc` 是在混合标签和强增强 batch 上统计的，不等同于原始训练集准确率。它低于 `test acc` 是正常现象。

如果需要记录更直观的干净训练集准确率，可以运行：

```bash
python my_final_CNN/train.py --amp --eval-tta --clean-train-eval
```

如果担心额外评估耗时，可以只评估部分 batch：

```bash
python my_final_CNN/train.py --amp --eval-tta --clean-train-eval --clean-train-max-batches 100
```

## 显存不足时

先减小 batch size：

```bash
python my_final_CNN/train.py --amp --eval-tta --batch-size 96
python my_final_CNN/train.py --amp --eval-tta --batch-size 64
```

如果仍然不足，可以使用 WRN-28-10 轻量版本：

```bash
python my_final_CNN/train.py --amp --eval-tta --depth 28 --widen-factor 10 --output-dir my_final_CNN/final_result/final_swrn28_10
```

## 生成图像和洞察分析

训练结束后重新生成训练曲线：

```bash
python my_final_CNN/plot_results.py
```

生成混淆矩阵、滤波器、Grad-CAM 等可视化：

```bash
python my_final_CNN/visualize.py --tta
```

生成 loss landscape：

```bash
python my_final_CNN/loss_landscape.py --max-batches 4 --points 21
```

## 项目要求对应关系

- 第 2 点：`model.py` 中包含 `Conv2d`、`AdaptiveAvgPool2d`、激活函数和 `Linear` 全连接层。
- 第 3 点：包含 BatchNorm、Dropout、残差连接、SE attention 和 stochastic depth。
- 第 4 点：可通过 `--depth`、`--widen-factor`、`--loss`、`--label-smoothing`、`--focal-gamma`、`--activation` 比较滤波器数量、损失函数和激活函数。
- 第 5 点：可通过 `--optimizer sgd|adamw|rmsprop` 使用 `torch.optim` 比较不同优化器。
- 第 6 点：`visualize.py` 与 `loss_landscape.py` 提供滤波器可视化、网络解释、误分类分析和损失景观。

## 建议报告口径

最终报告中可将本模型与此前模型对比：

```text
Classic CNN: 86.85%
ResNet-110: 93.40%
DenseNet-BC-100-24: 96.42%
WRN-28-10: 96.63%
FinalCNN SE-SD-WRN-40-10: 训练后填写最终结果
```

如果最终 300 epoch 的结果略低于预期，可以优先尝试以下更保守的配置：

```bash
python my_final_CNN/train.py --amp --eval-tta --activation relu --stochastic-depth-rate 0.05 --output-dir my_final_CNN/final_result/final_swrn40_10_relu_sd005
```
