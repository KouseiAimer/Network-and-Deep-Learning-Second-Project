# Classic DenseNet-BC on CIFAR-10

本目录实现 CIFAR-10 上的 DenseNet-BC。默认配置偏向精度，目标是冲到约 95% 测试准确率：

- DenseNet-BC-100-24
- depth = 100
- growth rate = 24
- compression = 0.5
- bottleneck layer: `1x1 Conv -> 3x3 Conv`
- 3 个 dense block，每个 block 16 个 dense layer
- 默认训练 200 epochs

如果显存压力较大，可以改成更轻的 `--growth-rate 12`。

## 文件说明

- `model.py`：定义 CIFAR 版 `DenseNetCIFAR` 和 `densenet_bc()`。
- `train.py`：训练、测试、保存权重和日志。
- `plot_results.py`：根据 `history.csv` 重新绘制曲线并生成 `summary.json`。
- `weights/densenet_bc_100_24/`：默认保存 `best.pt`、`last.pt` 等模型权重。权重文件已被 `.gitignore` 忽略，后续适合上传到网盘。
- `results/densenet_bc_100_24/`：默认保存 `history.csv`、`curves.png`、`summary.json`、`config.json`。

## 训练

在项目根目录运行：

```powershell
conda activate dl
python .\classic_DenseNet\train.py
```

默认配置：

- epochs: 200
- batch size: 64
- optimizer: SGD + momentum 0.9 + Nesterov
- initial learning rate: 0.1
- weight decay: `1e-4`
- scheduler: CosineAnnealingLR
- data augmentation: RandomCrop + RandomHorizontalFlip + Cutout
- label smoothing: 0.1

## 快速测试

```powershell
python .\classic_DenseNet\train.py --epochs 1 --subset 256 --batch-size 32 --lr 0.01 --num-workers 0 --eval-max-batches 2
```

## 显存不足时

先尝试减小 batch size：

```powershell
python .\classic_DenseNet\train.py --batch-size 32 --amp
```

如果仍然显存不足，可以使用轻量版本：

```powershell
python .\classic_DenseNet\train.py --growth-rate 12 --weights-dir .\classic_DenseNet\weights\densenet_bc_100_12 --results-dir .\classic_DenseNet\results\densenet_bc_100_12
```

## 进一步冲精度

默认配置已经比 ResNet-110 更偏精度。如果 200 epochs 结果略低于 95%，可以尝试：

```powershell
python .\classic_DenseNet\train.py --epochs 300
python .\classic_DenseNet\train.py --mixup-alpha 0.2
python .\classic_DenseNet\train.py --scheduler multistep --milestones 100 150
```

## 重新画图

```powershell
python .\classic_DenseNet\plot_results.py
```
