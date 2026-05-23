# PyramidNet + ShakeDrop Results

Training saves lightweight experiment artifacts here by default:

- `history.csv`: per-epoch loss, accuracy, learning rate, and runtime.
- `curves.png`: loss, accuracy, learning-rate, error, and generalization-gap curves.
- `summary.json`: best epoch and final metrics.
- `config.json`: training configuration and model parameter count.
- `visualizations/`: optional confusion matrix, per-class accuracy, confidence histogram, misclassified examples, and prediction CSV.

These files are suitable for GitHub and reports. Model checkpoints should be uploaded to a netdisk instead.
