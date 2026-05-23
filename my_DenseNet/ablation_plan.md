# Ablation Plan

This file lists focused experiments that satisfy Project 2 Task 1 requirements 4 and 5.

Run these after the main model if time permits. Each command saves to a separate folder under `final_result/ablations/`.

## Different Numbers of Filters

```powershell
python .\my_DenseNet\train.py --depth 190 --growth-rate 24 --output-dir .\my_DenseNet\final_result\ablations\growth24 --amp
python .\my_DenseNet\train.py --depth 190 --growth-rate 32 --output-dir .\my_DenseNet\final_result\ablations\growth32 --amp
python .\my_DenseNet\train.py --depth 190 --growth-rate 40 --output-dir .\my_DenseNet\final_result\ablations\growth40 --amp
```

## Different Loss Functions

```powershell
python .\my_DenseNet\train.py --loss ce --label-smoothing 0.1 --output-dir .\my_DenseNet\final_result\ablations\loss_ce_smooth --amp
python .\my_DenseNet\train.py --loss focal --focal-gamma 2.0 --label-smoothing 0.05 --output-dir .\my_DenseNet\final_result\ablations\loss_focal --amp
```

## Different Activations

```powershell
python .\my_DenseNet\train.py --activation relu --output-dir .\my_DenseNet\final_result\ablations\act_relu --amp
python .\my_DenseNet\train.py --activation silu --output-dir .\my_DenseNet\final_result\ablations\act_silu --amp
python .\my_DenseNet\train.py --activation gelu --output-dir .\my_DenseNet\final_result\ablations\act_gelu --amp
```

## Different Optimizers from `torch.optim`

```powershell
python .\my_DenseNet\train.py --optimizer sgd --lr 0.1 --output-dir .\my_DenseNet\final_result\ablations\optim_sgd --amp
python .\my_DenseNet\train.py --optimizer adamw --lr 0.001 --weight-decay 0.05 --output-dir .\my_DenseNet\final_result\ablations\optim_adamw --amp
python .\my_DenseNet\train.py --optimizer rmsprop --lr 0.01 --output-dir .\my_DenseNet\final_result\ablations\optim_rmsprop --amp
```

For a quick ablation preview, add:

```powershell
--epochs 60
```

For final reporting, prefer full-length runs or clearly mark them as short ablations.
