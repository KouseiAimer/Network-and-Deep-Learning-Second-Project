# Network and Deep Learning Second Project

This repository contains CIFAR-10 experiments for the second project of Neural Network and Deep Learning.

## Contents

- `classic_CNN/`: classic CNN baseline, training script, result record, and curves.
- `classic_ResNet-110/`: CIFAR ResNet-110 implementation, training script, results, and report.
- `classic_DenseNet/`: DenseNet-BC implementation targeting stronger CIFAR-10 accuracy.
- `classic_PyramidNet+ShakeDrop/`: advanced PyramidNet + ShakeDrop implementation with richer visualization tools.
- `classic_WideResNet/`: strong WRN-28-10 implementation with CutMix/Cutout/EMA training recipe.
- `my_DenseNet/`: final improved SE-DenseNet-BC network with ablations and interpretation scripts.
- `codes/VGG_BatchNorm/`: provided VGG/BatchNorm experiment code.
- `project_2_2026_zh.md`: Chinese translation of the project handout.

## Large Files

Datasets and model checkpoints are intentionally excluded from Git. Upload trained weights such as `best.pt` and `last.pt` to a netdisk and include the links in the final report.
