# Final Result Directory

All server-side outputs for `my_DenseNet` should be saved under this directory.

Default training writes to:

```text
my_DenseNet/final_result/final_se_densenet/
```

Expected files after training:

- `config.json`: full model and training configuration.
- `history.csv`: per-epoch loss, accuracy, learning rate, and time.
- `summary.json`: best epoch and final metrics.
- `curves.png`: training curves.
- `weights/best.pt`: best checkpoint, upload to netdisk rather than GitHub.
- `weights/last.pt`: final checkpoint, upload to netdisk rather than GitHub.
- `visualizations/`: optional interpretation figures from `visualize.py`.
- `loss_landscape/`: optional local loss landscape from `loss_landscape.py`.

When the server finishes training, copy the whole `final_result` folder back to the local project for analysis and report writing.
