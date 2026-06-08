# Ablation 自动化实验说明

这个目录用于围绕最终模型 `MyDenseNet / SE-DenseNet-BC-190-40` 做第一部分报告所需的消融实验、最终复跑和可视化。这里的代码参考了此前 DenseNet 实验的设置，但模型、训练、日志、消融和可视化逻辑都已经在 `Ablation/` 内重新编写，运行时不需要调用其他实验目录下的 Python 脚本。

## 文件作用

- `ablation.py`：自动运行一组 50 epoch 的短预算消融实验，并生成 `summary.csv` 和 `best_config.json`。
- `final.py`：读取 `best_config.json` 中由消融结果推荐的配置，再跑一次完整最终实验，默认 300 epoch。
- `visual_ablation.py`：读取消融实验结果，生成柱状图、学习曲线、按组最优表格和 Markdown 汇总。
- `visual_final.py`：对最终实验生成训练曲线、混淆矩阵、逐类准确率、置信度分布、误分类样例、第一层滤波器、Grad-CAM 和局部 loss landscape。

## 推荐运行顺序

在项目根目录运行，并先进入带有 PyTorch 的环境：

```bash
conda activate dl
python Ablation/ablation.py --epochs 50 --amp
python Ablation/visual_ablation.py
python Ablation/final.py --epochs 300 --amp
python Ablation/visual_final.py
```

如果只是想先检查代码流程，可以用很小的数据子集：

```bash
python Ablation/ablation.py --epochs 1 --subset 256 --eval-max-batches 2 --num-workers 0 --rerun
python Ablation/visual_ablation.py
python Ablation/final.py --epochs 1 --num-workers 0 --amp
python Ablation/visual_final.py --max-batches 2 --landscape-max-batches 1 --landscape-points 5
```

## 默认消融内容

默认 `--suite report` 会运行用于报告要求的核心消融：

| 组别 | 实验 |
|---|---|
| baseline | 默认 `SE-DenseNet-BC-190-40 + SiLU + CE + SGD` |
| capacity | `growth_rate=32`、`growth_rate=24`，对比不同滤波器数量 |
| activation | `ReLU`、`GELU`，对比默认 `SiLU` |
| loss | `Focal Loss + label smoothing 0.05`，对比默认交叉熵 |
| optimizer | `AdamW`、`RMSprop`，对比默认 SGD |

如果时间充足，可以运行更完整的组件消融：

```bash
python Ablation/ablation.py --suite full --epochs 50 --amp
```

`--suite full` 会额外加入：

- 去掉 SE attention；
- 去掉 stochastic depth；
- 去掉 classifier dropout；
- 关闭 CutMix 和 Cutout。

## 输出目录

消融实验默认输出到：

```text
Ablation/results/ablation/
```

每个实验都有独立子目录，例如：

```text
Ablation/results/ablation/capacity/capacity_growth32/
```

其中包含：

- `config.json`：该实验的训练配置；
- `history.csv`：逐 epoch 日志；
- `summary.json`：最佳准确率、错误率、最佳 epoch 等；
- `curves.png`：训练曲线；
- `weights/best.pt` 和 `weights/last.pt`：模型权重。

总表保存在：

```text
Ablation/results/ablation/summary.csv
Ablation/results/ablation/best_config.json
```

`best_config.json` 会记录两类信息：

- `best_single_experiment`：50 epoch 短消融中单个表现最好的实验；
- `recommended_config`：按 capacity、activation、loss、optimizer 等组别分别选择较优设置后组合出的推荐最终配置。

`final.py` 默认使用 `recommended_config` 进行完整复跑。

最终实验默认输出到：

```text
Ablation/results/final/best_from_ablation/
```

## 报告写法建议

最终报告中建议明确说明：

> 主模型结果来自完整 300 epoch 训练；消融实验使用 50 epoch 的统一短预算，目的是比较设计选择对早期收敛速度和泛化趋势的影响，而不是替代最终精度评测。

这样既能满足项目中“尝试不同滤波器数量、损失函数、激活函数、优化器”的要求，也不会把大量时间浪费在每个消融都完整训练 300 epoch 上。

## 常用参数

- `--epochs`：每个实验训练轮数。消融默认 50，最终默认 300。
- `--suite report|full`：消融套件规模，默认 `report`。
- `--rerun`：即使已有 `summary.json` 也重新运行。
- `--skip-existing`：默认行为，已有结果则跳过。
- `--subset` 和 `--eval-max-batches`：只用于快速测试，不建议用于正式报告。
- `--no-amp`：关闭混合精度。

权重文件通常较大，不建议上传 GitHub。最终提交报告时，把最终实验的 `weights/best.pt` 和 `weights/last.pt` 上传网盘并在报告中附链接。
