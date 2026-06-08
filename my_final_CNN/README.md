# 最终 CNN：FinalDenseNetV2

本目录现在包含两条路线：

- `v1`：`model.py` + `train.py`，即 SE-StochasticDepth WideResNet-40-10。服务器完整训练结果不理想，最佳测试准确率为 91.08%。
- `v2`：`model_v2.py` + `train_v2.py`，即基于实测最佳模型重塑的 SE-DenseNet-BC-190-40。推荐后续使用这一版作为最终 CNN。

## 为什么放弃 v1 WRN

此前 `classic_WideResNet` 的 WRN-28-10 已经达到 96.63%，但 `my_final_CNN` v1 同时引入更深 WRN-40-10、SE、stochastic depth 和 SiLU 后，训练明显受抑：

```text
classic WRN-28-10 best: 96.63%
my_final_CNN v1 SE-SD-WRN-40-10 best: 91.08%
```

从日志看，v1 在第 50 轮仅 77.29%，而经典 WRN-28-10 在第 50 轮已达 87.72%。这说明问题不是后期 TTA 或 checkpoint 选择，而是早期优化阶段就已经明显落后。由于 RandAugment、CutMix、EMA 等配方在 WRN-28-10 中已经成功，主要风险来自“更深 WRN + SE + stochastic depth + SiLU + dropout”的一次性叠加，导致有效残差分支被过度正则化，最终出现欠拟合。

## v2 默认模型

`FinalDenseNetV2` 回到已验证最强的 DenseNet 主线，并保留项目要求的组件与可调实验接口：

```text
FinalDenseNetV2 = SE-DenseNet-BC-190-40
depth = 190
growth_rate = 40
compression = 0.5
activation = SiLU
SE reduction = 16
stochastic depth rate = 0.2
classifier hidden dim = 512
classifier dropout = 0.2
trainable parameters = 26,763,454
```

它包含全连接层、二维卷积层、二维池化层、激活函数、BatchNorm、Dropout、残差式 dense connectivity、SE attention 和 stochastic depth。

## 文件说明

- `model_v2.py`：推荐最终网络，SE-DenseNet-BC-190-40。
- `train_v2.py`：推荐训练脚本，默认复现已验证的 DenseNet 配方，并增加 TTA、clean train eval、mixing 控制。
- `evaluate_v2.py`：单模型 TTA 评估和多 checkpoint/多 seed 概率平均集成。
- `model.py` / `train.py`：保留 v1 WRN，便于复现实验失败分析。
- `plot_results.py`：训练曲线与 `summary.json`。
- `visualize.py`：混淆矩阵、逐类准确率、置信度分布、误分类样例、滤波器和 Grad-CAM，已兼容 v1 WRN 与 v2 DenseNet。
- `loss_landscape.py`：局部一维 loss landscape，已兼容 v1 与 v2。
- `ablation_plan.md`：v2 消融实验命令。
- `final_result/`：服务器输出目录。

## 推荐训练命令

在项目根目录运行：

```bash
conda activate dl
python my_final_CNN/train_v2.py --amp --eval-tta
```

默认输出：

```text
my_final_CNN/final_result/final_densenet_v2_seed42/
```

如果你想最稳地冲击 97% 左右，建议在 20 至 40GB 显存服务器上跑 2 到 3 个 seed，然后使用 `evaluate_v2.py` 做概率平均：

```bash
python my_final_CNN/train_v2.py --amp --eval-tta --seed 42 --output-dir my_final_CNN/final_result/final_densenet_v2_seed42
python my_final_CNN/train_v2.py --amp --eval-tta --seed 3407 --output-dir my_final_CNN/final_result/final_densenet_v2_seed3407
python my_final_CNN/train_v2.py --amp --eval-tta --seed 2026 --output-dir my_final_CNN/final_result/final_densenet_v2_seed2026
```

集成评估：

```bash
python my_final_CNN/evaluate_v2.py --tta \
  --checkpoints \
  my_final_CNN/final_result/final_densenet_v2_seed42/weights/best.pt \
  my_final_CNN/final_result/final_densenet_v2_seed3407/weights/best.pt \
  my_final_CNN/final_result/final_densenet_v2_seed2026/weights/best.pt \
  --output my_final_CNN/final_result/final_densenet_v2_ensemble_summary.json
```

单模型报告应仍以单个 `best.pt` 为主；多 seed ensemble 可以作为额外强结果汇报。

## 生成分析图

训练后重新生成曲线：

```bash
python my_final_CNN/plot_results.py --history my_final_CNN/final_result/final_densenet_v2_seed42/history.csv --output my_final_CNN/final_result/final_densenet_v2_seed42/curves.png --summary my_final_CNN/final_result/final_densenet_v2_seed42/summary.json
```

生成混淆矩阵、滤波器和 Grad-CAM：

```bash
python my_final_CNN/visualize.py --tta \
  --checkpoint my_final_CNN/final_result/final_densenet_v2_seed42/weights/best.pt \
  --config my_final_CNN/final_result/final_densenet_v2_seed42/config.json \
  --output-dir my_final_CNN/final_result/final_densenet_v2_seed42/visualizations
```

生成 loss landscape：

```bash
python my_final_CNN/loss_landscape.py \
  --checkpoint my_final_CNN/final_result/final_densenet_v2_seed42/weights/best.pt \
  --config my_final_CNN/final_result/final_densenet_v2_seed42/config.json \
  --output-dir my_final_CNN/final_result/final_densenet_v2_seed42/loss_landscape \
  --max-batches 4 --points 21
```

## 项目要求对应

- 第 2 点：`model_v2.py` 包含 `Conv2d`、`AvgPool2d` / `AdaptiveAvgPool2d`、激活函数与 `Linear`。
- 第 3 点：包含 BatchNorm、Dropout、dense connectivity、SE attention 和 stochastic depth。
- 第 4 点：可通过 `--depth`、`--growth-rate`、`--loss`、`--label-smoothing`、`--focal-gamma`、`--activation` 比较滤波器数量、损失函数和激活函数。
- 第 5 点：可通过 `--optimizer sgd|adamw|rmsprop` 使用 `torch.optim` 比较优化器。
- 第 6 点：`visualize.py` 与 `loss_landscape.py` 提供滤波器可视化、误分类分析、Grad-CAM 和损失景观。

## 权重提交

`.pt` 权重文件不会上传到 GitHub。最终报告中请将 `weights/best.pt` 和 `weights/last.pt` 上传到网盘，并附上链接。
