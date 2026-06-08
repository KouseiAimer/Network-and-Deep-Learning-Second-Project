# Visual 可视化说明

本目录用于把基础 Batch Normalization 实验结果画成更接近论文/讲义风格的图。代码依赖 `Batch_Normalization/results` 中已有的 `history.csv`、`batch_losses.csv`、`best.pt` 和 `last.pt`，默认不重新训练模型。

## 文件说明

- `visualize_bn.py`：统一可视化脚本。
- `figures/`：运行后自动生成，保存图片。
- `cache/`：运行 3D loss surface 时自动生成，缓存 loss 网格，避免重复计算。

## 运行前准备

```bash
conda activate dl
cd Batch_Normalization/Visual
```

## 图 1：论文风格 Accuracy 曲线

这张图直接读取基础实验的 `history.csv`，画出类似参考图左半部分的训练/测试准确率对比。

```bash
python visualize_bn.py --mode accuracy --accuracy-lrs 1e-3 2e-3
```

输出：

```text
figures/paper_style_accuracy_panels.png
```

推荐报告解读：`lr=2e-3` 时无 BN 接近随机猜测，而 BN 仍能正常训练，说明 BN 扩大了可用学习率范围。

## 图 2：3D Loss Surface

这张图加载 `best.pt` 或 `last.pt`，在参数空间中选取两个 filter-normalized random directions，然后计算：

```text
L(w + alpha * d1 + beta * d2)
```

并画出 no BN 和 BN 的左右 3D surface 对比。

快速版本：

```bash
python visualize_bn.py --mode surface --surface-lr 1e-3 --checkpoint best --surface-grid 17 --surface-samples 512
```

更细版本：

```bash
python visualize_bn.py --mode surface --surface-lr 1e-3 --checkpoint best --surface-grid 25 --surface-samples 1024
```

输出：

```text
figures/loss_surface_3d_no_bn_vs_bn.png
cache/*.npz
```

注意：3D surface 是高维参数空间的二维切片，不是完整 loss landscape。不同随机方向会带来不同视觉形态，因此报告中应描述为“局部二维切片”。

## 图 3：Activation Distribution Ridgeline

这张图加载 checkpoint，对 CIFAR-10 验证集采样，然后在若干卷积层上收集激活分布。对 no BN 模型，默认采集卷积输出；对 BN 模型，默认采集对应 BatchNorm 层输出。

```bash
python visualize_bn.py --mode activation --activation-lr 1e-3 --checkpoint best --activation-samples 1024 --activation-layers 1 4 8
```

输出：

```text
figures/activation_distribution_ridgeline.png
figures/activation_distribution_stats.csv
```

推荐报告解读：如果 BN 的激活分布更集中、不同层之间尺度更一致，可以说明 BN 稳定了中间表征的分布。

## 一次性运行全部图

```bash
python visualize_bn.py --mode all --accuracy-lrs 1e-3 2e-3 --surface-grid 17 --surface-samples 512 --activation-samples 1024
```

如果只想快速生成最稳妥的报告图，建议先运行：

```bash
python visualize_bn.py --mode accuracy
python visualize_bn.py --mode activation --activation-samples 1024
```

3D surface 图最漂亮，但计算量最大，建议最后再跑。
