# Classic CNN Baseline

这是 CIFAR-10 第一个任务的传统 CNN baseline。网络结构使用经典的 `Conv2d -> ReLU -> MaxPool2d -> Fully Connected` 路线，不使用 BatchNorm 和残差连接，方便后续和 BN、VGG、ResNet 等模型做对照。

## 文件说明

- `model.py`：定义 `ClassicCNN` 和参数量统计函数。
- `train.py`：加载本地 CIFAR-10、训练、测试、保存 checkpoint 和训练曲线。
- `runs/baseline/`：训练后自动生成，包含 `best.pt`、`last.pt`、`history.csv` 和 `curves.png`。

## 运行方式

在项目根目录打开终端，并进入你的 conda 环境：

```powershell
conda activate dl
python .\classic_CNN\train.py
```

默认会读取本地数据集目录：

```text
data/cifar-10-batches-py
```

如果只想快速检查代码是否能跑通，可以先用小数据子集：

```powershell
conda activate dl
python .\classic_CNN\train.py --epochs 1 --subset 512 --batch-size 64 --num-workers 0
```

## 常用参数

```powershell
python .\classic_CNN\train.py --epochs 100 --batch-size 128 --lr 0.1
python .\classic_CNN\train.py --optimizer adam --lr 0.001
python .\classic_CNN\train.py --no-augment
python .\classic_CNN\train.py --amp
```

## Baseline 配置

- 数据增强：随机裁剪、随机水平翻转。
- 归一化：CIFAR-10 常用 mean/std。
- 损失函数：CrossEntropyLoss。
- 默认优化器：SGD + Momentum + Nesterov。
- 默认学习率策略：CosineAnnealingLR。
- 保存指标：每个 epoch 的 train/test loss 和 accuracy。
