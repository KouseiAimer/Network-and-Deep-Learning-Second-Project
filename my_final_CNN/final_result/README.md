# FinalCNN 结果目录

服务器训练产生的结果统一保存在本目录下。当前推荐的 v2 默认实验目录为：

```text
my_final_CNN/final_result/final_densenet_v2_seed42/
```

此前 `final_swrn40_10/` 为 v1 WideResNet 路线的实验结果，最佳测试准确率较低，主要用于失败分析。

建议保留并回传以下文件：

- `config.json`：训练配置和模型参数量。
- `history.csv`：每个 epoch 的 loss、accuracy、学习率和耗时。
- `summary.json`：最佳 epoch、最佳测试准确率和最终指标。
- `curves.png`：训练曲线。
- `visualizations/`：混淆矩阵、Grad-CAM、滤波器等图像。
- `loss_landscape/`：loss landscape CSV 和图像。

`.pt` 权重文件不会上传到 GitHub。最终提交时请将 `weights/best.pt` 和 `weights/last.pt` 上传到网盘。
