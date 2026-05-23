# Classic PyramidNet + ShakeDrop on CIFAR-10

本目录实现 CIFAR-10 上的 PyramidNet + ShakeDrop。它比普通 ResNet/DenseNet 更适合做高级 CNN 对比实验：

- PyramidNet 将通道数逐块平滑增加，而不是只在 stage 边界突然翻倍。
- ShakeDrop 对残差分支做随机前向/反向缩放，是一种专门面向深残差网络的强正则化方法。
- 默认配置为 `PyramidNet-110-a270 + ShakeDrop`，目标是作为高精度高级 CNN baseline。

## 文件说明

- `model.py`：定义 `PyramidNetCIFAR`、`ShakeDrop` 和 `pyramidnet_shakedrop()`。
- `train.py`：训练、测试、保存 checkpoint 和日志。
- `plot_results.py`：根据 `history.csv` 生成训练曲线和 `summary.json`。
- `visualize.py`：加载 `best.pt` 生成混淆矩阵、每类准确率、置信度直方图、误分类样例和预测 CSV。
- `weights/pyramidnet110_a270_shakedrop/`：默认保存 `best.pt`、`last.pt` 等权重。权重文件已被 `.gitignore` 忽略，后续上传网盘。
- `results/pyramidnet110_a270_shakedrop/`：默认保存 `history.csv`、`curves.png`、`summary.json`、`config.json` 和可视化结果。

## 训练

在项目根目录运行：

```powershell
conda activate dl
python .\classic_PyramidNet+ShakeDrop\train.py
```

默认配置：

- model: `PyramidNet-110-a270 + ShakeDrop`
- epochs: 300
- batch size: 64
- optimizer: SGD + momentum 0.9 + Nesterov
- initial learning rate: 0.1
- weight decay: `1e-4`
- scheduler: CosineAnnealingLR
- data augmentation: RandomCrop + RandomHorizontalFlip + Cutout
- label smoothing: 0.1
- final survival probability: 0.5

## 快速测试

完整默认模型较大。只检查代码流程时，建议使用小模型和少量 batch：

```powershell
python .\classic_PyramidNet+ShakeDrop\train.py --depth 20 --alpha 48 --epochs 1 --subset 128 --batch-size 32 --lr 0.01 --num-workers 0 --eval-max-batches 2 --weights-dir .\classic_PyramidNet+ShakeDrop\weights\smoke_test --results-dir .\classic_PyramidNet+ShakeDrop\results\smoke_test
```

## 显存不足时

优先尝试 AMP 和更小 batch size：

```powershell
python .\classic_PyramidNet+ShakeDrop\train.py --batch-size 32 --amp
```

如果仍然显存不足，可以使用轻量版本：

```powershell
python .\classic_PyramidNet+ShakeDrop\train.py --depth 110 --alpha 84 --weights-dir .\classic_PyramidNet+ShakeDrop\weights\pyramidnet110_a84_shakedrop --results-dir .\classic_PyramidNet+ShakeDrop\results\pyramidnet110_a84_shakedrop
```

## 重新画训练曲线

```powershell
python .\classic_PyramidNet+ShakeDrop\plot_results.py
```

## 生成更多结果图

训练完成后，使用最佳 checkpoint 生成更多分析图：

```powershell
python .\classic_PyramidNet+ShakeDrop\visualize.py
```

会输出：

- `confusion_matrix.png`
- `per_class_accuracy.png`
- `confidence_histogram.png`
- `misclassified_examples.png`
- `predictions.csv`
