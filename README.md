# Social Media Content Moderation Using Noisy Labels

End-to-end PyTorch + HuggingFace Transformers implementation for Hinglish hate-speech moderation with noisy and weak labels.

## Project Structure

```text
DSAI_project/
|-- combined_hate_speech_dataset.csv  # Full Hinglish dataset, used by default
|-- config.py                         # Hyperparameters and paths
|-- main.py                           # Full BERT pipeline entry point
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
python main.py --no_noise
python main.py --noise_rate 0.4 --noise_type symmetric
python main.py --dataset_path combined_hate_speech_dataset.csv --text_column text --label_column hate_label
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
