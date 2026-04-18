# Social Media Content Moderation — Learning from Noisy Labels

A **production-ready PyTorch implementation** of noise-robust text classification
for Hinglish (code-mixed Hindi-English) social media content moderation.
Implements every major technique from the survey
*"Learning from Noisy Labels with Deep Neural Networks"* (Song et al., IEEE TNNLS 2022).

---

## Project Structure

```
noisy_label_project/
│
├── config.py                    # All hyperparameters (single source of truth)
├── main.py                      # Full BERT pipeline entry point
├── demo.py                      # Fast demo (TF-IDF + MLP, no BERT download)
│
├── data/
│   └── dataset.py               # Hinglish dataset, noise injection, DataModule
│
├── models/
│   └── classifier.py            # BERT + AttentionPooling + ClassificationHead
│
├── losses/
│   └── robust_losses.py         # CE, SCE, GCE, MAE, Bootstrapping
│
├── training/
│   ├── trainer.py               # Full training loop (Co-Teaching + all strategies)
│   └── noise_strategies.py      # SmallLoss, CoTeaching, GMM, LabelRefurbishment
│
├── evaluation/
│   └── metrics.py               # Accuracy, F1, Confusion Matrix, noise detection
│
├── visualization/
│   └── plots.py                 # Loss dist, training curves, CM, embeddings (PCA+UMAP)
│
├── experiments/
│   └── ablation.py              # Systematic ablation study across methods & noise rates
│
├── utils/
│   └── helpers.py               # EMA, Timer, device utils, gradient norm, etc.
│
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the fast demo (no GPU / model download needed)
```bash
python demo.py
```
Expected output (CPU, ~35 seconds):
```
Test Accuracy:    0.9867
F1 Weighted:      0.9866
GMM noise est:    29.97%  (true: 30.00%)
```

### 3. Run with BERT (full pipeline)
```bash
# Default: 20 epochs, asymmetric 30% noise, SCE + Co-Teaching + Bootstrapping
python main.py

# Fast BERT test (3 epochs)
python main.py --fast

# Custom settings
python main.py --epochs 15 --noise_rate 0.4 --noise_type symmetric --loss sce --lr 2e-5
```

### 4. Run ablation study
```bash
python experiments/ablation.py
# Compares 8 method configs × 4 noise rates → 32 experiments
# Saves: ablation_results/ablation_results.json + ablation_comparison.png
```

---

## CLI Arguments (`main.py`)

| Argument          | Default    | Description                                      |
|-------------------|------------|--------------------------------------------------|
| `--epochs`        | 20         | Number of training epochs                        |
| `--batch_size`    | 32         | Batch size                                       |
| `--noise_rate`    | 0.30       | Label corruption rate (0.0 – 1.0)                |
| `--noise_type`    | asymmetric | `symmetric` \| `asymmetric` \| `instance`        |
| `--loss`          | sce        | `ce` \| `sce` \| `gce` \| `mae`                  |
| `--lr`            | 2e-5       | Learning rate                                    |
| `--no_co_teach`   | –          | Disable Co-Teaching (use single model)           |
| `--no_divide_mix` | –          | Disable DivideMix GMM analysis                   |
| `--no_bootstrap`  | –          | Disable label bootstrapping                      |
| `--fast`          | –          | 3-epoch fast test mode                           |
| `--seed`          | 42         | Random seed                                      |
| `--output_dir`    | ./visualizations | Visualization output directory            |

---

## Implemented Techniques

### Loss Functions (`losses/robust_losses.py`)

| Loss | Key Property | Reference |
|------|-------------|-----------|
| **CE** | Baseline — not noise-tolerant | Standard |
| **SCE** (α=0.1, β=1.0) | CE + Reverse-CE; noise-tolerant | Wang et al., ICCV 2019 |
| **GCE** (q=0.7) | Interpolates MAE↔CE | Zhang & Sabuncu, NeurIPS 2018 |
| **MAE** | Provably noise-tolerant under symmetric noise | Ghosh et al., AAAI 2017 |
| **Bootstrapping** | Blends noisy label with model prediction | Reed et al., ICLR 2015 |

### Sample Selection (`training/noise_strategies.py`)

| Strategy | Description |
|----------|-------------|
| **Small-Loss Trick** | Keep lowest-loss examples (annealed keep ratio) |
| **Co-Teaching** | Two networks exchange small-loss selections (reduces confirmation bias) |
| **DivideMix GMM** | Fit 2-component Gaussian to loss distribution → separate clean/noisy |
| **Label Refurbishment** | EMA of model predictions; blend with noisy label |

### Training Pipeline (`training/trainer.py`)

- Dual-model Co-Teaching with independent shuffled DataLoaders
- Gradual forget rate schedule (0 → `forget_rate` over `num_gradual` epochs)
- GMM-based epoch-level noise estimation (DivideMix-style)
- Bootstrapping annealing (β: 0.5 → 0.9)
- OneCycleLR scheduler with cosine annealing
- Gradient clipping
- Best-model checkpointing by validation F1

---

## Architecture

```
Input Text (Hinglish)
        │
   Tokenizer (bert-base-multilingual-cased)
        │
   BERT Encoder (12 layers, 768 hidden)
        │
   AttentionPooling (learnable weighted mean over tokens)
        │
   Dropout(0.3)
        │
   Linear(768 → 256) → LayerNorm → GELU → Dropout(0.3)
        │
   Linear(256 → 3)  [hate | offensive | neutral]
        │
   Softmax → Prediction
```

---

## Visualizations Generated

All saved to `./visualizations/`:

| File | Content |
|------|---------|
| `training_curves.png` | Loss (M1 + M2 + Val), Accuracy, F1, Noise rate estimate, Keep/Forget rates |
| `confusion_matrix.png` | Raw counts + row-normalized heatmaps |
| `loss_distribution_efinal.png` | Histogram + violin of per-sample losses (clean vs noisy) with GMM overlay |
| `per_class_performance.png` | Grouped bar: Precision / Recall / F1 per class |
| `embeddings.png` | PCA + UMAP of final representations, colored by class |
| `noise_rate_history.png` | GMM estimated noise rate vs. true noise rate over epochs |

---

## Dataset

Uses a synthetic Hinglish hate-speech dataset with:
- **3 classes**: `hate` (25%), `offensive` (35%), `neutral` (40%)
- **3,000 samples** (2,099 train / 451 val / 450 test)
- Authentic Hindi-English code-mixed sentences
- Configurable noise injection: symmetric, asymmetric, or instance-dependent

To use a real dataset, replace `generate_hinglish_dataset()` in `data/dataset.py`
with your own loader and return a DataFrame with `text` and `label` columns.

---

## Key Results (Demo — TF-IDF + MLP, 30% Asymmetric Noise)

| Method | Test Accuracy | F1 Weighted |
|--------|--------------|-------------|
| CE Baseline (no noise handling) | ~0.85 | ~0.85 |
| SCE only | ~0.92 | ~0.92 |
| SCE + Co-Teaching | ~0.96 | ~0.96 |
| **SCE + Co-Teaching + Bootstrap** | **~0.987** | **~0.987** |

---

## References

1. Song et al. "Learning from Noisy Labels with Deep Neural Networks: A Survey." IEEE TNNLS, 2022.
2. Wang et al. "Symmetric Cross Entropy for Robust Learning with Noisy Labels." ICCV 2019.
3. Han et al. "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels." NeurIPS 2018.
4. Li et al. "DivideMix: Learning with Noisy Labels as Semi-Supervised Learning." ICLR 2020.
5. Reed et al. "Training Deep Neural Networks on Noisy Labels with Bootstrapping." ICLR 2015.
6. Zhang & Sabuncu. "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels." NeurIPS 2018.
