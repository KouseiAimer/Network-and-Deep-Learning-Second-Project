# My DenseNet Final Network

This folder contains the final improved network for CIFAR-10 Task 1. The model is an enhanced DenseNet-BC with:

- 2D convolution layers
- 2D pooling layers
- activation functions
- fully connected classifier layers
- BatchNorm
- Dropout / DropPath
- Squeeze-and-Excitation channel attention
- stochastic depth
- configurable loss functions, activations, filter counts, and optimizers
- visualization scripts for curves, confusion matrix, filters, Grad-CAM, and loss landscape

The default final model is:

```text
MyDenseNet = SE-DenseNet-BC-190-40
depth = 190
growth_rate = 40
compression = 0.5
activation = SiLU
SE reduction = 16
stochastic depth rate = 0.2
classifier hidden dim = 512
```

This is intentionally stronger than `classic_DenseNet/DenseNet-BC-100-24`. It is designed for your server with about 20 GB GPU memory.

## Files

- `model.py`: improved DenseNet architecture.
- `train.py`: training, evaluation, checkpointing, EMA, Mixup/CutMix, optimizer and loss selection.
- `plot_results.py`: plots training curves and writes `summary.json`.
- `visualize.py`: confusion matrix, per-class accuracy, confidence histogram, first-layer filters, Grad-CAM, misclassified examples.
- `loss_landscape.py`: local 1D loss landscape around the best checkpoint.
- `ablation_plan.md`: commands for required ablation experiments.
- `final_result/`: all server outputs should be saved here.

## Recommended Server Training Command

Run from the project root:

```powershell
conda activate dl
python .\my_DenseNet\train.py --amp
```

On Linux server:

```bash
conda activate dl
python my_DenseNet/train.py --amp
```

Default outputs:

```text
my_DenseNet/final_result/final_se_densenet/
```

## If GPU Memory Is Tight

Try smaller batch size first:

```bash
python my_DenseNet/train.py --batch-size 32 --amp
```

If still too large, use a smaller growth rate:

```bash
python my_DenseNet/train.py --growth-rate 32 --batch-size 64 --amp --output-dir my_DenseNet/final_result/final_se_densenet_g32
```

## Quick Smoke Test

This only checks code flow and should not be used for reporting:

```powershell
python .\my_DenseNet\train.py --depth 40 --growth-rate 12 --epochs 1 --subset 128 --batch-size 16 --lr 0.01 --num-workers 0 --eval-max-batches 2 --output-dir .\my_DenseNet\final_result\smoke_test
```

## Generate Figures After Training

Training automatically creates `curves.png`, but after copying server outputs back locally you can regenerate it:

```powershell
python .\my_DenseNet\plot_results.py
```

Generate interpretation figures:

```powershell
python .\my_DenseNet\visualize.py
```

Generate a local loss landscape:

```powershell
python .\my_DenseNet\loss_landscape.py --max-batches 4 --points 21
```

## Project Requirement Mapping

- Requirement 2: Conv2d, AvgPool2d/AdaptiveAvgPool2d, activations, and fully connected layers are included.
- Requirement 3: BatchNorm, Dropout, stochastic depth, and SE attention are included.
- Requirement 4: `--growth-rate`, `--depth`, `--loss`, `--label-smoothing`, `--focal-gamma`, and `--activation` support filter/loss/activation experiments.
- Requirement 5(a): `--optimizer sgd|adamw|rmsprop` uses optimizers from `torch.optim`.
- Requirement 6: `plot_results.py`, `visualize.py`, and `loss_landscape.py` provide training curves, filter visualization, Grad-CAM, confusion matrix, and loss landscape.

## What to Send Back from the Server

After training, copy back:

```text
my_DenseNet/final_result/
```

The `.pt` weights should not be uploaded to GitHub. Upload `best.pt` and `last.pt` to a netdisk and put the links in the final project report.
