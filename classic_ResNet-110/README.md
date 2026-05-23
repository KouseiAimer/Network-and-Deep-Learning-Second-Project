# Classic ResNet-110 on CIFAR-10

本目录复现 CIFAR-10 版本的 ResNet-110。模型采用原始 ResNet 论文中用于 CIFAR 的结构：

- 深度满足 `6n + 2 = 110`，因此 `n = 18`。
- 初始卷积通道数为 16。
- 三个 residual stage 的通道数分别为 16、32、64。
- 每个 stage 包含 18 个 BasicBlock，每个 BasicBlock 有两个 `3x3` 卷积。
- 默认 shortcut 使用 CIFAR ResNet 论文中的 Option A：下采样时用 stride 切片和 zero-padding，不引入额外投影参数。

## 文件说明

- `model.py`：定义 CIFAR 版 `ResNetCIFAR` 和 `resnet110()`。
- `train.py`：训练、测试、保存权重、记录日志。
- `plot_results.py`：根据 `history.csv` 重新绘制训练曲线并生成 `summary.json`。
- `weights/resnet110/`：训练后保存 `best.pt`、`last.pt` 等模型权重。权重文件已被本目录 `.gitignore` 忽略，后续适合上传到网盘。
- `results/resnet110/`：训练后保存 `history.csv`、`curves.png`、`summary.json`、`config.json`，适合随报告和 GitHub 一起保留。

## 训练

在项目根目录运行：

```powershell
conda activate dl
python .\classic_ResNet-110\train.py
```

默认配置为 200 epochs：

- batch size: 128
- optimizer: SGD + momentum 0.9 + Nesterov
- initial learning rate: 0.1
- weight decay: `1e-4`
- scheduler: MultiStepLR，在第 100 和 150 个 epoch 将学习率乘以 0.1
- data augmentation: RandomCrop + RandomHorizontalFlip

## 快速测试

```powershell
python .\classic_ResNet-110\train.py --epochs 1 --subset 512 --batch-size 64 --lr 0.01 --num-workers 0
```

## 重新画图

训练结束后，或修改图样式后，可以重新生成曲线和摘要：

```powershell
python .\classic_ResNet-110\plot_results.py
```

## 常用变体

```powershell
python .\classic_ResNet-110\train.py --amp
python .\classic_ResNet-110\train.py --scheduler cosine
python .\classic_ResNet-110\train.py --shortcut-type B
python .\classic_ResNet-110\train.py --resume .\classic_ResNet-110\weights\resnet110\last.pt
```
