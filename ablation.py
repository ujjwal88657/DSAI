"""
experiments/ablation.py
Ablation study runner: systematically compares all noise-handling
strategies and loss functions across different noise rates.
Generates a combined results table and comparison plots.

Run:  python experiments/ablation.py
"""

import os
import sys
import json
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import generate_hinglish_dataset, apply_noise
from robust_losses import (
    CrossEntropyLoss, SymmetricCrossEntropyLoss,
    GeneralizedCrossEntropyLoss, MAELoss, BootstrappingLoss,
    compute_per_sample_loss,
)
from noise_strategies import (
    CoTeaching, GaussianMixtureNoiseSeparator,
    SmallLossTrick, LabelRefurbishmentStore,
)
from metrics import compute_metrics, compute_loss
from helpers import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight MLP (same as demo.py)
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return {"logits": self.net(x)}


class TFIDFDS(torch.utils.data.Dataset):
    def __init__(self, X, labels, is_noisy=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.noisy = torch.tensor(is_noisy if is_noisy is not None else [0]*len(labels), dtype=torch.long)
        self.idx = torch.arange(len(labels))

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"features": self.X[i], "label": self.labels[i],
                "is_noisy": self.noisy[i], "index": self.idx[i]}


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    X_train, y_train_noisy, X_val, y_val, X_test, y_test,
    noisy_mask,
    loss_type: str = "sce",
    use_co_teaching: bool = True,
    use_bootstrapping: bool = True,
    num_classes: int = 3,
    num_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-3,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    set_seed(seed)

    train_ds = TFIDFDS(X_train, y_train_noisy, is_noisy=noisy_mask)
    val_ds   = TFIDFDS(X_val, y_val)
    test_ds  = TFIDFDS(X_test, y_test)

    loader1 = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    loader2 = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader  = DataLoader(val_ds,  batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    input_dim = X_train.shape[1]
    model1 = MLP(input_dim, num_classes).to(device)
    model2 = MLP(input_dim, num_classes).to(device) if use_co_teaching else None

    # Loss function
    loss_map = {
        "ce":  CrossEntropyLoss(num_classes, reduction="none"),
        "sce": SymmetricCrossEntropyLoss(num_classes, 0.1, 1.0, reduction="none"),
        "gce": GeneralizedCrossEntropyLoss(num_classes, q=0.7, reduction="none"),
        "mae": MAELoss(num_classes, reduction="none"),
    }
    loss_fn   = loss_map[loss_type]
    boot_fn   = BootstrappingLoss(num_classes, beta=0.8, reduction="none")
    co_teach  = CoTeaching(forget_rate=0.20, num_gradual=5, total_epochs=num_epochs)

    opt1 = torch.optim.AdamW(model1.parameters(), lr=lr, weight_decay=1e-4)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=lr, weight_decay=1e-4) if model2 else None
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=num_epochs)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=num_epochs) if opt2 else None

    bootstrap_start = 5
    history = {"val_acc": [], "val_f1": []}

    for epoch in range(num_epochs):
        model1.train()
        if model2: model2.train()

        for b1, b2 in zip(loader1, loader2 if use_co_teaching else loader1):
            f1 = b1["features"].to(device); l1 = b1["label"].to(device)
            f2 = b2["features"].to(device); l2 = b2["label"].to(device)

            if use_co_teaching and model2 is not None:
                with torch.no_grad():
                    lo1 = model1(f1)["logits"]; lo2 = model2(f2)["logits"]
                ps1 = compute_per_sample_loss(lo1, l1)
                ps2 = compute_per_sample_loss(lo2, l2)
                ci1 = co_teach.select_indices(ps1, epoch)
                ci2 = co_teach.select_indices(ps2, epoch)

                # Model1 trains on model2's clean selection
                opt1.zero_grad()
                out1 = model1(f2[ci2])
                fn = boot_fn if epoch >= bootstrap_start and use_bootstrapping else loss_fn
                loss1 = fn(out1["logits"], l2[ci2]).mean()
                loss1.backward(); nn.utils.clip_grad_norm_(model1.parameters(), 1.0); opt1.step()

                # Model2 trains on model1's clean selection
                opt2.zero_grad()
                out2 = model2(f1[ci1])
                loss2 = fn(out2["logits"], l1[ci1]).mean()
                loss2.backward(); nn.utils.clip_grad_norm_(model2.parameters(), 1.0); opt2.step()
            else:
                # Standard training (no co-teaching)
                opt1.zero_grad()
                out1 = model1(f1)
                fn = boot_fn if epoch >= bootstrap_start and use_bootstrapping else loss_fn
                loss = fn(out1["logits"], l1).mean()
                loss.backward(); nn.utils.clip_grad_norm_(model1.parameters(), 1.0); opt1.step()

        sched1.step()
        if sched2: sched2.step()

        # Validate
        model1.eval()
        preds, probs, labels = [], [], []
        with torch.no_grad():
            for b in val_loader:
                out = model1(b["features"].to(device))
                p = F.softmax(out["logits"], dim=1)
                preds.extend(p.argmax(1).cpu().numpy())
                probs.extend(p.cpu().numpy())
                labels.extend(b["label"].numpy())
        import numpy as np
        m = compute_metrics(np.array(preds), np.array(labels))
        history["val_acc"].append(m["accuracy"])
        history["val_f1"].append(m["f1_weighted"])

    # Test evaluation
    model1.eval()
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for b in test_loader:
            out = model1(b["features"].to(device))
            p = F.softmax(out["logits"], dim=1)
            preds.extend(p.argmax(1).cpu().numpy())
            probs.extend(p.cpu().numpy())
            labels.extend(b["label"].numpy())
    import numpy as np
    preds_arr = np.array(preds)
    probs_arr = np.array(probs)
    labels_arr = np.array(labels)
    test_metrics = compute_metrics(preds_arr, labels_arr)
    test_metrics["loss"] = compute_loss(probs_arr, labels_arr)
    test_metrics["history"] = history
    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Ablation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(output_dir: str = "./ablation_results"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Ablation] Device: {device}")
    print("[Ablation] Generating dataset...")

    df = generate_hinglish_dataset(n_samples=3000, seed=42)
    train_val_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.15/0.85, stratify=train_val_df["label"], random_state=42)
    val_df = val_df.copy(); val_df["label"] = val_df["original_label"]
    test_df = test_df.copy(); test_df["label"] = test_df["original_label"]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_all = vectorizer.fit_transform(train_df["text"]).toarray()
    X_val = vectorizer.transform(val_df["text"]).toarray()
    X_test = vectorizer.transform(test_df["text"]).toarray()
    y_val = val_df["label"].values
    y_test = test_df["label"].values
    y_clean = train_df["original_label"].values

    NUM_CLASSES = 3
    EPOCHS = 10

    # ── Experiment grid ──────────────────────────────────────────────────────
    noise_rates = [0.0, 0.15, 0.30, 0.45]
    configs = [
        {"name": "CE (Baseline)",            "loss": "ce",  "co_teach": False, "bootstrap": False},
        {"name": "GCE",                       "loss": "gce", "co_teach": False, "bootstrap": False},
        {"name": "MAE",                       "loss": "mae", "co_teach": False, "bootstrap": False},
        {"name": "SCE",                       "loss": "sce", "co_teach": False, "bootstrap": False},
        {"name": "SCE + Bootstrap",           "loss": "sce", "co_teach": False, "bootstrap": True},
        {"name": "CE + Co-Teaching",          "loss": "ce",  "co_teach": True,  "bootstrap": False},
        {"name": "SCE + Co-Teaching",         "loss": "sce", "co_teach": True,  "bootstrap": False},
        {"name": "SCE + Co-Teaching + Boot",  "loss": "sce", "co_teach": True,  "bootstrap": True},
    ]

    all_results = {}
    summary_rows = []

    print(f"\n[Ablation] Running {len(configs)} configs × {len(noise_rates)} noise rates "
          f"= {len(configs)*len(noise_rates)} experiments\n")

    for nr in noise_rates:
        print(f"\n{'─'*60}")
        print(f"  NOISE RATE = {nr:.0%}")
        print(f"{'─'*60}")

        # Inject noise at this rate
        if nr > 0:
            noisy_df = apply_noise(train_df.copy(), "asymmetric", nr, NUM_CLASSES, seed=42)
            y_train_noisy = noisy_df["label"].values
            noisy_mask = noisy_df["is_noisy"].values
        else:
            y_train_noisy = y_clean.copy()
            noisy_mask = np.zeros(len(y_clean), dtype=int)

        for cfg in configs:
            t0 = time.time()
            print(f"  [{cfg['name']:40s}] ", end="", flush=True)

            try:
                metrics = run_experiment(
                    X_train_all, y_train_noisy, X_val, y_val, X_test, y_test,
                    noisy_mask=noisy_mask,
                    loss_type=cfg["loss"],
                    use_co_teaching=cfg["co_teach"],
                    use_bootstrapping=cfg["bootstrap"],
                    num_classes=NUM_CLASSES,
                    num_epochs=EPOCHS,
                    device=device,
                )
                acc = metrics["accuracy"]
                f1  = metrics["f1_weighted"]
                elapsed = time.time() - t0
                print(f"Acc={acc:.4f}  F1={f1:.4f}  ({elapsed:.1f}s)")

                key = f"{cfg['name']}__nr{int(nr*100)}"
                all_results[key] = {
                    "config": cfg["name"],
                    "noise_rate": nr,
                    "accuracy": acc,
                    "f1_macro": metrics["f1_macro"],
                    "f1_weighted": f1,
                    "per_class_f1": metrics["per_class_f1"],
                }
                summary_rows.append({
                    "Method": cfg["name"],
                    "Noise": f"{nr:.0%}",
                    "Accuracy": f"{acc:.4f}",
                    "F1_W": f"{f1:.4f}",
                })

            except Exception as e:
                print(f"ERROR: {e}")
                summary_rows.append({
                    "Method": cfg["name"],
                    "Noise": f"{nr:.0%}",
                    "Accuracy": "ERROR",
                    "F1_W": "ERROR",
                })

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = os.path.join(output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Ablation] Results saved to {results_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY TABLE")
    print(f"{'='*80}")
    header = f"{'Method':<42}{'Noise':>8}{'Accuracy':>12}{'F1 Weighted':>14}"
    print(header)
    print("─" * 80)
    prev_noise = None
    for row in summary_rows:
        if row["Noise"] != prev_noise:
            if prev_noise is not None:
                print("─" * 80)
            prev_noise = row["Noise"]
        print(f"{row['Method']:<42}{row['Noise']:>8}{row['Accuracy']:>12}{row['F1_W']:>14}")
    print("=" * 80)

    # ── Plot comparison ───────────────────────────────────────────────────────
    try:
        _plot_ablation(all_results, configs, noise_rates, output_dir)
    except Exception as e:
        print(f"[Ablation] Plot error: {e}")

    return all_results


def _plot_ablation(results, configs, noise_rates, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Ablation Study: Noise-Robust Methods vs. Noise Rate", fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    for metric_idx, (metric_key, metric_label) in enumerate([
        ("accuracy", "Accuracy"), ("f1_weighted", "F1 Weighted")
    ]):
        ax = axes[metric_idx]
        for cfg, color in zip(configs, colors):
            vals = []
            for nr in noise_rates:
                key = f"{cfg['name']}__nr{int(nr*100)}"
                if key in results:
                    vals.append(results[key].get(metric_key, None))
                else:
                    vals.append(None)

            valid_x = [nr for nr, v in zip(noise_rates, vals) if v is not None]
            valid_y = [v for v in vals if v is not None]

            if valid_y:
                ax.plot([x*100 for x in valid_x], valid_y,
                        marker="o", linewidth=2, label=cfg["name"],
                        color=color, markersize=6)

        ax.set_xlabel("Noise Rate (%)", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"{metric_label} vs. Noise Rate", fontsize=12)
        ax.set_ylim([0.5, 1.02])
        ax.set_xticks([nr*100 for nr in noise_rates])
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7, loc="lower left", frameon=True)

    fig.tight_layout()
    path = os.path.join(output_dir, "ablation_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Ablation] Plot saved: {path}")


if __name__ == "__main__":
    run_ablation(output_dir="./ablation_results")
