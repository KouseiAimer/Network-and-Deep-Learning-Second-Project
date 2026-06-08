# FinalDenseNetV2 消融实验计划

以下命令围绕 `train_v2.py` 设计，用于满足 project 第 1 部分第 4、5、6 点。建议每个实验使用不同 `--output-dir`，训练完成后回传 `config.json`、`history.csv`、`summary.json` 和曲线图。

## 1. 滤波器数量 / 网络容量

```bash
python my_final_CNN/train_v2.py --amp --eval-tta --depth 190 --growth-rate 32 --output-dir my_final_CNN/final_result/ablations_v2/capacity_d190_k32
python my_final_CNN/train_v2.py --amp --eval-tta --depth 190 --growth-rate 40 --output-dir my_final_CNN/final_result/ablations_v2/capacity_d190_k40
python my_final_CNN/train_v2.py --amp --eval-tta --depth 250 --growth-rate 40 --batch-size 48 --output-dir my_final_CNN/final_result/ablations_v2/capacity_d250_k40
```

`d190_k40` 是已验证的稳健配置；`d250_k40` 需要更多显存和时间，适合 40GB GPU 尝试。

## 2. 损失函数与正则化

```bash
python my_final_CNN/train_v2.py --amp --eval-tta --loss ce --label-smoothing 0.0 --output-dir my_final_CNN/final_result/ablations_v2/loss_ce_ls0
python my_final_CNN/train_v2.py --amp --eval-tta --loss ce --label-smoothing 0.1 --output-dir my_final_CNN/final_result/ablations_v2/loss_ce_ls01
python my_final_CNN/train_v2.py --amp --eval-tta --loss focal --focal-gamma 2.0 --label-smoothing 0.05 --output-dir my_final_CNN/final_result/ablations_v2/loss_focal
```

## 3. 激活函数

```bash
python my_final_CNN/train_v2.py --amp --eval-tta --activation relu --output-dir my_final_CNN/final_result/ablations_v2/act_relu
python my_final_CNN/train_v2.py --amp --eval-tta --activation silu --output-dir my_final_CNN/final_result/ablations_v2/act_silu
python my_final_CNN/train_v2.py --amp --eval-tta --activation gelu --output-dir my_final_CNN/final_result/ablations_v2/act_gelu
```

## 4. torch.optim 优化器

```bash
python my_final_CNN/train_v2.py --amp --eval-tta --optimizer sgd --lr 0.1 --weight-decay 1e-4 --output-dir my_final_CNN/final_result/ablations_v2/optim_sgd
python my_final_CNN/train_v2.py --amp --eval-tta --optimizer adamw --lr 0.001 --weight-decay 0.05 --output-dir my_final_CNN/final_result/ablations_v2/optim_adamw
python my_final_CNN/train_v2.py --amp --eval-tta --optimizer rmsprop --lr 0.01 --weight-decay 1e-4 --output-dir my_final_CNN/final_result/ablations_v2/optim_rmsprop
```

## 5. 组件贡献

```bash
python my_final_CNN/train_v2.py --amp --eval-tta --se-reduction 0 --output-dir my_final_CNN/final_result/ablations_v2/no_se
python my_final_CNN/train_v2.py --amp --eval-tta --stochastic-depth-rate 0.0 --output-dir my_final_CNN/final_result/ablations_v2/no_stochastic_depth
python my_final_CNN/train_v2.py --amp --eval-tta --classifier-dropout 0.0 --output-dir my_final_CNN/final_result/ablations_v2/no_classifier_dropout
python my_final_CNN/train_v2.py --amp --eval-tta --no-cutout --mix-mode none --output-dir my_final_CNN/final_result/ablations_v2/no_cutmix_cutout
```

## 6. 多 seed 与集成

```bash
python my_final_CNN/train_v2.py --amp --eval-tta --seed 42 --output-dir my_final_CNN/final_result/final_densenet_v2_seed42
python my_final_CNN/train_v2.py --amp --eval-tta --seed 3407 --output-dir my_final_CNN/final_result/final_densenet_v2_seed3407
python my_final_CNN/train_v2.py --amp --eval-tta --seed 2026 --output-dir my_final_CNN/final_result/final_densenet_v2_seed2026

python my_final_CNN/evaluate_v2.py --tta \
  --checkpoints \
  my_final_CNN/final_result/final_densenet_v2_seed42/weights/best.pt \
  my_final_CNN/final_result/final_densenet_v2_seed3407/weights/best.pt \
  my_final_CNN/final_result/final_densenet_v2_seed2026/weights/best.pt \
  --output my_final_CNN/final_result/final_densenet_v2_ensemble_summary.json
```

## 7. 洞察可视化

```bash
python my_final_CNN/plot_results.py --history my_final_CNN/final_result/final_densenet_v2_seed42/history.csv --output my_final_CNN/final_result/final_densenet_v2_seed42/curves.png --summary my_final_CNN/final_result/final_densenet_v2_seed42/summary.json

python my_final_CNN/visualize.py --tta \
  --checkpoint my_final_CNN/final_result/final_densenet_v2_seed42/weights/best.pt \
  --config my_final_CNN/final_result/final_densenet_v2_seed42/config.json \
  --output-dir my_final_CNN/final_result/final_densenet_v2_seed42/visualizations

python my_final_CNN/loss_landscape.py \
  --checkpoint my_final_CNN/final_result/final_densenet_v2_seed42/weights/best.pt \
  --config my_final_CNN/final_result/final_densenet_v2_seed42/config.json \
  --output-dir my_final_CNN/final_result/final_densenet_v2_seed42/loss_landscape \
  --max-batches 4 --points 21
```
