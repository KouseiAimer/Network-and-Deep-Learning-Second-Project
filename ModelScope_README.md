# 神经网络与深度学习 Project 2 模型权重

本 ModelScope 仓库保存复旦大学《神经网络与深度学习》Project 2 实验中产生的模型权重。GitHub 仓库只保存代码、配置、训练日志、图像和报告材料；`.pt` 权重文件由于体积较大，不上传到 GitHub。

GitHub 代码仓库：

```text
https://github.com/KouseiAimer/Network-and-Deep-Learning-Second-Project
```

## 路径对应关系

ModelScope 中的权重路径与 GitHub 仓库中的相对路径保持一致。例如：

```text
ModelScope:
Ablation/results/final/best_from_ablation/weights/best.pt

GitHub 对应代码与结果:
Ablation/ablation.py
Ablation/final.py
Ablation/results/final/best_from_ablation/config.json
Ablation/results/final/best_from_ablation/summary.json
```

因此，下载任意权重后，可以直接回到 GitHub 中相同目录查看模型结构、训练配置、曲线和实验摘要。

## 主要权重说明

| ModelScope 权重目录 | GitHub 对应模型/脚本 | 说明 |
|---|---|---|
| `Ablation/results/final/best_from_ablation/weights/` | `Ablation/ablation.py`, `Ablation/final.py` | 最终消融后模型，SE-DenseNet-BC-190-24 + GELU，最佳测试准确率 97.15%。 |
| `Ablation/results/ablation/*/*/weights/` | `Ablation/ablation.py` | 50 epoch 消融实验权重，覆盖 growth rate、activation、loss、optimizer 等对照。 |
| `classic_CNN/runs/baseline/` | `classic_CNN/model.py`, `classic_CNN/train.py` | 传统 CNN baseline。 |
| `classic_ResNet-110/weights/resnet110/` | `classic_ResNet-110/model.py`, `classic_ResNet-110/train.py` | CIFAR ResNet-110 baseline。 |
| `classic_DenseNet/weights/densenet_bc_100_24/` | `classic_DenseNet/model.py`, `classic_DenseNet/train.py` | DenseNet-BC-100-24 baseline。 |
| `classic_PyramidNet+ShakeDrop/weights/pyramidnet110_a270_shakedrop/` | `classic_PyramidNet+ShakeDrop/model.py`, `classic_PyramidNet+ShakeDrop/train.py` | PyramidNet-110-a270 + ShakeDrop 对照模型。 |
| `classic_WideResNet/final_result/wrn28_10/weights/` | `classic_WideResNet/model.py`, `classic_WideResNet/train.py` | WRN-28-10 强 baseline。 |
| `my_DenseNet/final_result/final_se_densenet/weights/` | `my_DenseNet/model.py`, `my_DenseNet/train.py` | 原最终 SE-DenseNet-BC-190-40 模型，最佳测试准确率 96.68%。 |
| `my_final_CNN/final_result/final_swrn40_10/weights/` | `my_final_CNN/model.py`, `my_final_CNN/train.py` | SE-StochasticDepth-WRN-40-10 失败分析模型。 |
| `Batch_Normalization/results/*/` | `Batch_Normalization/models.py`, `Batch_Normalization/run_bn_experiments.py` | VGG-A 有/无 BN 的基础实验权重。 |
| `Batch_Normalization/Enhanced/results/*/*/` | `Batch_Normalization/Enhanced/enhanced_models.py`, `Batch_Normalization/Enhanced/run_enhanced_experiments.py` | BN 学习率敏感性与梯度实验权重。 |

每个实验目录通常包含：

- `best.pt`：测试集表现最好的 checkpoint；
- `last.pt`：最后一个 epoch 结束时的 checkpoint；
- GitHub 中同路径附近的 `config.json`、`summary.json`、`history.csv`：用于复现实验设置与指标。

## 最终模型

本项目第一部分最终采用：

```text
SE-DenseNet-BC-190-24
growth_rate = 24
activation = GELU
loss = CrossEntropy + label smoothing 0.1
optimizer = SGD + momentum + Nesterov
augmentation = AutoAugment + Cutout + CutMix
EMA = 0.999
best test accuracy = 97.15%
best test error = 2.85%
```

最终权重：

```text
Ablation/results/final/best_from_ablation/weights/best.pt
Ablation/results/final/best_from_ablation/weights/last.pt
```

对应 GitHub 代码和结果：

```text
Ablation/ablation.py
Ablation/final.py
Ablation/visual_final.py
Ablation/results/final/best_from_ablation/config.json
Ablation/results/final/best_from_ablation/summary.json
```
