# Network and Deep Learning Second Project

This repository contains CIFAR-10 experiments for the second project of Neural Network and Deep Learning.

## Contents

- `classic_CNN/`: classic CNN baseline, training script, result record, and curves.
- `classic_ResNet-110/`: CIFAR ResNet-110 implementation, training script, results, and report.
- `classic_DenseNet/`: DenseNet-BC implementation targeting stronger CIFAR-10 accuracy.
- `classic_PyramidNet+ShakeDrop/`: advanced PyramidNet + ShakeDrop implementation with richer visualization tools.
- `classic_WideResNet/`: strong WRN-28-10 implementation with CutMix/Cutout/EMA training recipe.
- `my_DenseNet/`: final improved SE-DenseNet-BC network with ablations and interpretation scripts.
- `my_final_CNN/`: final CNN workspace; v2 uses SE-DenseNet-BC with TTA/ensemble evaluation after the WRN v1 trial underfit.
- `codes/VGG_BatchNorm/`: provided VGG/BatchNorm experiment code.
- `project_2_2026_zh.md`: Chinese translation of the project handout.

## Large Files

Datasets and model checkpoints are intentionally excluded from Git. All trained checkpoints are uploaded to ModelScope:

```text
https://modelscope.cn/models/KouseiAimer/Network-and-Deep-Learning-Second-Project
```

The ModelScope repository preserves the same relative paths as this GitHub repository, so a checkpoint such as `Ablation/results/final/best_from_ablation/weights/best.pt` on ModelScope corresponds to the model code and configs under the same path in this repo.
