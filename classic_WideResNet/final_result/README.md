# WideResNet Final Result Directory

All training outputs are saved here by default.

Default run:

```text
classic_WideResNet/final_result/wrn28_10/
```

Expected files after training:

- `config.json`: model and training configuration.
- `history.csv`: per-epoch loss, accuracy, learning rate, and time.
- `summary.json`: best epoch and final metrics.
- `curves.png`: training curves.
- `weights/best.pt`: best checkpoint, upload to netdisk instead of GitHub.
- `weights/last.pt`: final checkpoint, upload to netdisk instead of GitHub.
- `visualizations/`: optional figures from `visualize.py`.

After training on the server, copy the whole `final_result` folder back for analysis and report writing.
