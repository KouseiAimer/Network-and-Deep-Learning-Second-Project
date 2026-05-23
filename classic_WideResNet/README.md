# Classic WideResNet on CIFAR-10

This folder implements a strong and stable WideResNet baseline for CIFAR-10.

Default model:

```text
WRN-28-10
depth = 28
widen_factor = 10
dropout = 0.3
activation = ReLU
```

The training recipe is stronger than a plain baseline:

- RandomCrop + RandomHorizontalFlip
- RandAugment by default
- Cutout
- CutMix by default, Mixup optional
- label smoothing
- EMA evaluation
- SGD + momentum + Nesterov
- cosine learning-rate schedule
- AMP support for server training

All outputs are saved under:

```text
classic_WideResNet/final_result/
```

## Files

- `model.py`: WRN implementation, with optional SE and stochastic depth.
- `train.py`: training, evaluation, checkpointing, EMA, Mixup/CutMix, optimizer and loss selection.
- `plot_results.py`: plots training curves and writes `summary.json`.
- `visualize.py`: confusion matrix, per-class accuracy, confidence histogram, first convolution filters, misclassified examples.
- `final_result/`: default result folder.

## Recommended Training

Run from the project root.

Windows:

```powershell
conda activate dl
python .\classic_WideResNet\train.py --amp
```

Linux server:

```bash
conda activate dl
python classic_WideResNet/train.py --amp
```

Default output folder:

```text
classic_WideResNet/final_result/wrn28_10/
```

## If GPU Memory Is Tight

First reduce batch size:

```bash
python classic_WideResNet/train.py --batch-size 64 --amp
```

For a lighter WRN:

```bash
python classic_WideResNet/train.py --widen-factor 8 --batch-size 128 --amp --output-dir classic_WideResNet/final_result/wrn28_8
```

## Stronger Variants To Try

Enable SE attention:

```bash
python classic_WideResNet/train.py --se-reduction 16 --amp --output-dir classic_WideResNet/final_result/wrn28_10_se
```

Use stochastic depth:

```bash
python classic_WideResNet/train.py --stochastic-depth-rate 0.1 --amp --output-dir classic_WideResNet/final_result/wrn28_10_sd
```

Try 300 epochs:

```bash
python classic_WideResNet/train.py --epochs 300 --amp --output-dir classic_WideResNet/final_result/wrn28_10_e300
```

## Quick Smoke Test

This only checks code flow:

```powershell
python .\classic_WideResNet\train.py --depth 16 --widen-factor 2 --epochs 1 --subset 128 --batch-size 16 --lr 0.01 --num-workers 0 --eval-max-batches 2 --output-dir .\classic_WideResNet\final_result\smoke_test
```

## Generate Figures After Training

Training automatically writes `curves.png`. To regenerate:

```powershell
python .\classic_WideResNet\plot_results.py
```

Generate additional visualizations:

```powershell
python .\classic_WideResNet\visualize.py
```

## Notes for Report

WideResNet satisfies the project requirements:

- Conv2d, pooling, activations, and fully connected layers are included.
- BatchNorm and Dropout are included; optional SE and stochastic depth are available.
- `--widen-factor`, `--loss`, and `--activation` support filter/loss/activation experiments.
- `--optimizer sgd|adamw|rmsprop` satisfies the torch.optim optimizer comparison requirement.
- `plot_results.py` and `visualize.py` provide training curves, error curves, filters, confusion matrix, and misclassification analysis.
