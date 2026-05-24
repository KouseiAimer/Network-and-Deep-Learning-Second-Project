# my_final_CNN 消融实验计划

以下命令用于满足 project 第 1 部分第 4、5、6 点的实验要求。建议每个实验使用不同 `--output-dir`，训练完成后将 `history.csv`、`summary.json` 和曲线图回传本地。

## 1. 滤波器数量 / 网络容量

```bash
python my_final_CNN/train.py --amp --eval-tta --depth 28 --widen-factor 10 --output-dir my_final_CNN/final_result/ablations/capacity_wrn28_10
python my_final_CNN/train.py --amp --eval-tta --depth 40 --widen-factor 8 --output-dir my_final_CNN/final_result/ablations/capacity_wrn40_8
python my_final_CNN/train.py --amp --eval-tta --depth 40 --widen-factor 10 --output-dir my_final_CNN/final_result/ablations/capacity_wrn40_10
```

## 2. 损失函数与正则化

```bash
python my_final_CNN/train.py --amp --eval-tta --loss ce --label-smoothing 0.0 --output-dir my_final_CNN/final_result/ablations/loss_ce_ls0
python my_final_CNN/train.py --amp --eval-tta --loss ce --label-smoothing 0.1 --output-dir my_final_CNN/final_result/ablations/loss_ce_ls01
python my_final_CNN/train.py --amp --eval-tta --loss focal --focal-gamma 2.0 --label-smoothing 0.05 --output-dir my_final_CNN/final_result/ablations/loss_focal
```

## 3. 激活函数

```bash
python my_final_CNN/train.py --amp --eval-tta --activation relu --output-dir my_final_CNN/final_result/ablations/act_relu
python my_final_CNN/train.py --amp --eval-tta --activation silu --output-dir my_final_CNN/final_result/ablations/act_silu
python my_final_CNN/train.py --amp --eval-tta --activation gelu --output-dir my_final_CNN/final_result/ablations/act_gelu
```

## 4. torch.optim 优化器

```bash
python my_final_CNN/train.py --amp --eval-tta --optimizer sgd --lr 0.1 --weight-decay 5e-4 --output-dir my_final_CNN/final_result/ablations/optim_sgd
python my_final_CNN/train.py --amp --eval-tta --optimizer adamw --lr 0.001 --weight-decay 0.05 --output-dir my_final_CNN/final_result/ablations/optim_adamw
python my_final_CNN/train.py --amp --eval-tta --optimizer rmsprop --lr 0.01 --weight-decay 5e-4 --output-dir my_final_CNN/final_result/ablations/optim_rmsprop
```

## 5. 组件贡献

```bash
python my_final_CNN/train.py --amp --eval-tta --se-reduction 0 --output-dir my_final_CNN/final_result/ablations/no_se
python my_final_CNN/train.py --amp --eval-tta --stochastic-depth-rate 0.0 --output-dir my_final_CNN/final_result/ablations/no_stochastic_depth
python my_final_CNN/train.py --amp --eval-tta --no-cutout --mix-mode none --output-dir my_final_CNN/final_result/ablations/no_cutmix_cutout
```

## 6. 洞察可视化

训练完成后运行：

```bash
python my_final_CNN/plot_results.py
python my_final_CNN/visualize.py --tta
python my_final_CNN/loss_landscape.py --max-batches 4 --points 21
```

报告中建议展示 `curves.png`、`confusion_matrix.png`、`first_conv_filters.png`、`gradcam_examples.png` 和 `loss_landscape_1d.png`。
