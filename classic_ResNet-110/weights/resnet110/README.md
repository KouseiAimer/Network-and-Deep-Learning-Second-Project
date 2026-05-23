# ResNet-110 Weights

Training saves model checkpoints here by default:

- `best.pt`: checkpoint with the best test accuracy.
- `last.pt`: checkpoint from the latest epoch.
- `epoch_NNN.pt`: optional periodic checkpoints when `--save-every N` is used.

These checkpoint files are ignored by `.gitignore`; upload them to a netdisk when preparing the final submission.
