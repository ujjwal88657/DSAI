# Social Media Content Moderation Using Noisy Labels

End-to-end PyTorch + HuggingFace Transformers implementation for Hinglish hate-speech moderation with noisy and weak labels.

## Project Structure

```text
DSAI_project/
|-- combined_hate_speech_dataset.csv  # Full Hinglish dataset, used by default
|-- config.py                         # Hyperparameters and paths
|-- main.py                           # Full BERT pipeline entry point
|-- analysis.py                       # Post-training metrics and visualization runner
|-- dataset.py                        # CSV loading, preprocessing, tokenization
|-- classifier.py                     # BERT classifier
|-- robust_losses.py                  # CE, SCE, GCE, MAE, bootstrapping
|-- trainer.py                        # Co-Teaching training loop
|-- noise_strategies.py               # Small-loss, GMM, label refurbishment
|-- metrics.py                        # Accuracy, precision, recall, F1, CM
|-- plots.py                          # Required visualizations
|-- demo.py                           # Lightweight TF-IDF + MLP demo
|-- ablation.py                       # Ablation runner
|-- helpers.py                        # Utility helpers
`-- requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run The Full Dataset

The default configuration reads the complete `combined_hate_speech_dataset.csv` file. It does not sample or truncate the dataset.

```bash
python main.py
```

Useful variants:

```bash
python main.py --fast
python main.py --epochs 10 --batch_size 16
python main.py --device auto
python main.py --device cpu
python main.py --no_noise
python main.py --noise_rate 0.4 --noise_type symmetric
python main.py --dataset_path combined_hate_speech_dataset.csv --text_column text --label_column hate_label
```

The default training length is now `10` epochs. You can still override it from the command line with `--epochs`.

## Generate Metrics And Visualizations

After training creates `checkpoints/best_model.pt`, run:

```bash
python analysis.py
```

This evaluates the saved model and writes analysis artifacts to `./analysis_outputs/`:

- `metrics_all_splits.json`
- `metrics_summary.csv`
- `per_class_metrics.csv`
- `classification_reports.txt`
- `train_predictions.csv`, `val_predictions.csv`, `test_predictions.csv`
- `train_loss_diagnostics.csv`
- confusion matrix
- per-class performance chart
- training curves
- loss distribution with GMM diagnostics
- noise-rate history
- PCA/UMAP embedding visualization

Useful variants:

```bash
python analysis.py --output_dir ./visualizations
python analysis.py --skip_embeddings
python analysis.py --splits val test
python analysis.py --checkpoint ./checkpoints/best_model.pt --device cpu
```

## Current Dataset Defaults

The bundled CSV has:

- `29,550` rows
- text column: `text`
- label column: `hate_label`
- labels: `0` and `1`
- class names: `non_hate`, `hate`

The data module performs a stratified `70/15/15` train/validation/test split over the full file. Synthetic noise, when enabled, is injected into the training split only. Validation and test labels remain clean for evaluation.

## Implemented Methods

- Symmetric Cross Entropy (SCE)
- Small-loss sample selection
- Co-Teaching with two BERT classifiers
- DivideMix-style loss GMM separation
- Bootstrapping / pseudo-label correction
- Optional noise injection: symmetric, asymmetric, instance-dependent

## Outputs

Training artifacts are written to:

- `checkpoints/best_model.pt`
- `logs/training_log.json`
- `logs/test_results.json`
- `visualizations/training_curves.png`
- `visualizations/confusion_matrix.png`
- `visualizations/loss_distribution_efinal.png`
- `visualizations/per_class_performance.png`
- `visualizations/embeddings.png`
- `visualizations/noise_rate_history.png`

## Notes

`bert-base-multilingual-cased` is the default model and will be downloaded by HuggingFace Transformers on the first run. Full Co-Teaching uses two BERT models, so GPU training is strongly recommended.

### Kaggle P100 CUDA Error

If Kaggle reports a Tesla P100 and then fails with:

```text
CUDA error: no kernel image is available for execution on the device
```

the installed PyTorch wheel does not include kernels for the P100 compute capability (`sm_60`). This is an environment mismatch, not a model/trainer bug. The training script now checks this before model training starts and prints the GPU architecture and PyTorch CUDA architecture list.

Best fixes:

```bash
# Easiest on Kaggle: switch the accelerator from P100 to T4/V100/A100.

# Or run without CUDA, which is much slower for full BERT:
python main.py --device cpu

# Or let the script choose CPU/MPS if CUDA is unusable:
python main.py --device auto
```

If you must use P100, install a PyTorch build that includes `sm_60`, then restart the notebook/kernel before running training again.
