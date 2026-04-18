"""
visualization/plots.py
All required visualizations:
  1. Loss distribution (per-sample, clean vs. noisy)
  2. Training curves (loss + F1 over epochs)
  3. Confusion matrix (normalized and raw)
  4. Embedding visualization (PCA + UMAP)
  5. Noise rate estimation over epochs
  6. Per-class performance bar chart
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional
from sklearn.decomposition import PCA


# -----------------------------------------------------------------------
# Style configuration
# -----------------------------------------------------------------------

PALETTE = {
    "hate": "#D62728",
    "offensive": "#FF7F0E",
    "neutral": "#2CA02C",
    "clean_loss": "#1F77B4",
    "noisy_loss": "#D62728",
    "model1": "#2E75B6",
    "model2": "#ED7D31",
    "train": "#1F4E79",
    "val": "#C00000",
}

def _setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "grid.color": "white",
        "grid.linewidth": 0.8,
    })

_setup_style()


def _save(fig, output_dir: str, name: str, dpi: int = 150, fmt: str = "png"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Viz] Saved: {path}")
    return path


# -----------------------------------------------------------------------
# 1. Loss Distribution Plot
# -----------------------------------------------------------------------

def plot_loss_distribution(
    per_sample_losses: np.ndarray,
    is_noisy: Optional[np.ndarray] = None,
    output_dir: str = "./visualizations",
    epoch: Optional[int] = None,
    gmm_params: Optional[Dict] = None,
    dpi: int = 150,
) -> str:
    """
    Histogram of per-sample losses, optionally colored by clean/noisy ground truth.
    If gmm_params provided, overlay fitted Gaussian components.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    title_suffix = f" (Epoch {epoch})" if epoch is not None else ""

    # ---- Left: full distribution ----
    ax = axes[0]
    if is_noisy is not None:
        clean_losses = per_sample_losses[is_noisy == 0]
        noisy_losses = per_sample_losses[is_noisy == 1]
        ax.hist(clean_losses, bins=50, alpha=0.65,
                color=PALETTE["clean_loss"], label=f"Clean (n={len(clean_losses)})", density=True)
        ax.hist(noisy_losses, bins=50, alpha=0.65,
                color=PALETTE["noisy_loss"], label=f"Noisy (n={len(noisy_losses)})", density=True)
        ax.legend(frameon=True)
    else:
        ax.hist(per_sample_losses, bins=60, color=PALETTE["model1"], alpha=0.8, density=True)

    # Overlay GMM Gaussians
    if gmm_params:
        x_range = np.linspace(per_sample_losses.min(), per_sample_losses.max(), 300)
        from scipy.stats import norm
        for comp, (mu, sig, pi, col) in enumerate([
            (gmm_params.get("mu_clean", 0), gmm_params.get("sigma_clean", 1),
             gmm_params.get("pi_clean", 0.5), PALETTE["clean_loss"]),
            (gmm_params.get("mu_noisy", 2), gmm_params.get("sigma_noisy", 1),
             1 - gmm_params.get("pi_clean", 0.5), PALETTE["noisy_loss"]),
        ]):
            y = pi * norm.pdf(x_range, mu, sig)
            label = "GMM Clean" if comp == 0 else "GMM Noisy"
            ax.plot(x_range, y, "--", color=col, linewidth=2, label=label)
        ax.legend(frameon=True, fontsize=9)

    ax.set_title(f"Per-Sample Loss Distribution{title_suffix}")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.4)

    # ---- Right: violin plot ----
    ax2 = axes[1]
    if is_noisy is not None:
        data = [clean_losses, noisy_losses]
        parts = ax2.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
        for pc_col, pc in zip([PALETTE["clean_loss"], PALETTE["noisy_loss"]], parts["bodies"]):
            pc.set_facecolor(pc_col)
            pc.set_alpha(0.7)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["Clean", "Noisy"])
    else:
        ax2.violinplot([per_sample_losses], positions=[0], showmeans=True, showmedians=True)
        ax2.set_xticks([0])
        ax2.set_xticklabels(["All samples"])

    ax2.set_title(f"Loss Violin{title_suffix}")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.4, axis="y")

    fig.suptitle("Loss Distribution Analysis — DivideMix / GMM Separation", fontsize=14, y=1.01)
    fig.tight_layout()
    return _save(fig, output_dir, f"loss_distribution_e{epoch or 'final'}", dpi=dpi)


# -----------------------------------------------------------------------
# 2. Training Curves
# -----------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List],
    output_dir: str = "./visualizations",
    dpi: int = 150,
) -> str:
    """Plot loss and metric curves over epochs for both models."""
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Dynamics — Noise-Robust Co-Teaching Pipeline", fontsize=15, y=1.01)

    # ---- Loss curves ----
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color=PALETTE["model1"], linewidth=2, label="Train (Model 1)")
    ax.plot(epochs, history["train_loss_m2"], color=PALETTE["model2"], linewidth=2,
            label="Train (Model 2)", linestyle="--")
    ax.plot(epochs, history["val_loss"], color=PALETTE["val"], linewidth=2, label="Validation")
    ax.set_title("Loss over Epochs")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(frameon=True); ax.grid(True, alpha=0.4)

    # ---- Accuracy ----
    ax = axes[0, 1]
    ax.plot(epochs, history["val_acc"], color=PALETTE["model1"], linewidth=2.5, marker="o", markersize=4)
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1]); ax.grid(True, alpha=0.4)

    # ---- F1 Score ----
    ax = axes[0, 2]
    ax.plot(epochs, history["val_f1"], color=PALETTE["model2"], linewidth=2.5, marker="s", markersize=4)
    ax.set_title("Validation F1 (Weighted)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.set_ylim([0, 1]); ax.grid(True, alpha=0.4)

    # ---- Estimated noise rate ----
    ax = axes[1, 0]
    nr_vals = [v for v in history.get("estimated_noise_rate", []) if v is not None]
    if nr_vals:
        nr_epochs = list(range(1, len(nr_vals) + 1))
        ax.plot(nr_epochs, nr_vals, color="#9467BD", linewidth=2)
        ax.axhline(y=0.30, color="red", linestyle=":", linewidth=1.5, label="True noise rate")
        ax.set_title("Estimated Noise Rate (GMM)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Noise Rate")
        ax.set_ylim([0, 0.7]); ax.legend(); ax.grid(True, alpha=0.4)
    else:
        ax.text(0.5, 0.5, "No GMM data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Estimated Noise Rate (GMM)")

    # ---- Keep / forget rate ----
    ax = axes[1, 1]
    kr = history.get("keep_ratio", [])
    fr = history.get("forget_rate", [])
    if kr:
        ax.plot(epochs, kr, color=PALETTE["train"], linewidth=2, label="Keep ratio")
    if fr:
        ax.plot(epochs, fr, color=PALETTE["val"], linewidth=2, linestyle="--", label="Forget rate")
    ax.set_title("Selection Rates over Epochs")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Rate")
    ax.set_ylim([0, 1.05]); ax.legend(); ax.grid(True, alpha=0.4)

    # ---- Epoch timing ----
    ax = axes[1, 2]
    et = history.get("epoch_time", [])
    if et:
        ax.bar(epochs, et, color=PALETTE["model1"], alpha=0.7)
        ax.set_title("Epoch Duration (seconds)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Time (s)")
        ax.grid(True, alpha=0.4, axis="y")
    else:
        ax.set_title("Epoch Duration")
        ax.text(0.5, 0.5, "No timing data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    return _save(fig, output_dir, "training_curves", dpi=dpi)


# -----------------------------------------------------------------------
# 3. Confusion Matrix
# -----------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: str = "./visualizations",
    title: str = "Confusion Matrix",
    dpi: int = 150,
) -> str:
    """Plot both raw and normalized confusion matrices side by side."""
    cm = np.array(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14)

    for ax, data, fmt, title_suffix, cmap in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Raw Counts", "Normalized (Row %)"],
        ["Blues", "YlOrRd"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap=cmap,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor="white",
            ax=ax, square=True, cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(title_suffix, fontsize=12)
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    return _save(fig, output_dir, "confusion_matrix", dpi=dpi)


# -----------------------------------------------------------------------
# 4. Embedding Visualization
# -----------------------------------------------------------------------

def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_dir: str = "./visualizations",
    method: str = "both",     # "pca" | "umap" | "both"
    sample_size: int = 500,
    dpi: int = 150,
    is_noisy: Optional[np.ndarray] = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
) -> str:
    """
    2D visualization of BERT embeddings using PCA and/or UMAP.
    Points colored by class label; optionally marked by noise status.
    """
    # Subsample for speed
    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        if is_noisy is not None:
            is_noisy = is_noisy[idx]

    colors = [PALETTE.get(cn, f"C{i}") for i, cn in enumerate(class_names)]
    label_colors = [colors[l] for l in labels]

    n_plots = 2 if method == "both" else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle("BERT Embedding Visualization", fontsize=15, y=1.01)
    plot_idx = 0

    # ---- PCA ----
    if method in ("pca", "both"):
        pca = PCA(n_components=2, random_state=42)
        coords_pca = pca.fit_transform(embeddings)
        ax = axes[plot_idx]
        _scatter_ax(ax, coords_pca, labels, label_colors, class_names, colors, is_noisy)
        var = pca.explained_variance_ratio_
        ax.set_title(f"PCA (Var: {var[0]:.1%} + {var[1]:.1%} = {sum(var):.1%})", fontsize=12)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        plot_idx += 1

    # ---- UMAP ----
    if method in ("umap", "both"):
        ax = axes[plot_idx]
        try:
            import umap
            reducer = umap.UMAP(
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                n_components=2,
                random_state=42,
            )
            coords_umap = reducer.fit_transform(embeddings)
            _scatter_ax(ax, coords_umap, labels, label_colors, class_names, colors, is_noisy)
            ax.set_title("UMAP", fontsize=12)
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        except ImportError:
            ax.text(0.5, 0.5, "umap-learn not installed.\nRun: pip install umap-learn",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11, color="red")
            ax.set_title("UMAP (unavailable)")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, output_dir, "embeddings", dpi=dpi)


def _scatter_ax(ax, coords, labels, label_colors, class_names, colors, is_noisy=None):
    """Helper: scatter plot with legend and optional noisy markers."""
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        mask = labels == lbl
        name = class_names[lbl] if lbl < len(class_names) else str(lbl)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=colors[lbl] if lbl < len(colors) else f"C{lbl}",
            label=name, alpha=0.7, s=25, edgecolors="none",
        )

    # Mark noisy samples with X
    if is_noisy is not None:
        noisy_mask = is_noisy == 1
        if noisy_mask.any():
            ax.scatter(
                coords[noisy_mask, 0], coords[noisy_mask, 1],
                marker="x", c="black", s=40, linewidths=0.8,
                label="Noisy label", alpha=0.5, zorder=5,
            )

    ax.legend(fontsize=9, frameon=True, loc="best", markerscale=1.5)


# -----------------------------------------------------------------------
# 5. Per-Class Performance Bar Chart
# -----------------------------------------------------------------------

def plot_per_class_performance(
    metrics: Dict,
    class_names: List[str],
    output_dir: str = "./visualizations",
    dpi: int = 150,
) -> str:
    """Grouped bar chart of Precision, Recall, F1 per class."""
    x = np.arange(len(class_names))
    width = 0.25

    precision = metrics.get("per_class_precision", [0] * len(class_names))
    recall = metrics.get("per_class_recall", [0] * len(class_names))
    f1 = metrics.get("per_class_f1", [0] * len(class_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_p = ax.bar(x - width, precision, width, label="Precision", color="#2E75B6", alpha=0.85)
    bars_r = ax.bar(x, recall, width, label="Recall", color="#ED7D31", alpha=0.85)
    bars_f = ax.bar(x + width, f1, width, label="F1", color="#70AD47", alpha=0.85)

    def _label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    _label_bars(bars_p)
    _label_bars(bars_r)
    _label_bars(bars_f)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylim([0, 1.15])
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision, Recall, and F1-Score", fontsize=13)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.4, axis="y")

    # Add overall metrics as text
    acc = metrics.get("accuracy", 0)
    f1_w = metrics.get("f1_weighted", 0)
    ax.text(0.98, 0.98, f"Overall Acc: {acc:.3f}\nF1 Weighted: {f1_w:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F4FD", edgecolor="#2E75B6"))

    fig.tight_layout()
    return _save(fig, output_dir, "per_class_performance", dpi=dpi)


# -----------------------------------------------------------------------
# 6. Noise Rate Estimation History
# -----------------------------------------------------------------------

def plot_noise_estimation_history(
    estimated_rates: List[float],
    true_rate: float,
    output_dir: str = "./visualizations",
    dpi: int = 150,
) -> str:
    """Line plot of GMM-estimated noise rate vs. true noise rate over epochs."""
    epochs = list(range(1, len(estimated_rates) + 1))
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, estimated_rates, color="#9467BD", linewidth=2.5,
            marker="o", markersize=5, label="GMM Estimate")
    ax.axhline(y=true_rate, color="#D62728", linewidth=2, linestyle="--",
               label=f"True Noise Rate ({true_rate:.0%})")
    ax.fill_between(epochs, estimated_rates, true_rate, alpha=0.15, color="#9467BD")

    ax.set_xlabel("Epoch"); ax.set_ylabel("Noise Rate")
    ax.set_ylim([0, max(0.7, max(estimated_rates) + 0.05)])
    ax.set_title("GMM Noise Rate Estimation vs. True Rate", fontsize=13)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return _save(fig, output_dir, "noise_rate_history", dpi=dpi)


# -----------------------------------------------------------------------
# Master visualization runner
# -----------------------------------------------------------------------

def run_all_visualizations(
    history: Dict,
    test_metrics: Dict,
    per_sample_losses: Optional[np.ndarray],
    is_noisy: Optional[np.ndarray],
    embeddings: Optional[np.ndarray],
    true_labels: Optional[np.ndarray],
    cfg,
) -> List[str]:
    """Run all visualizations and return list of saved file paths."""
    output_dir = cfg.viz.output_dir
    dpi = cfg.viz.dpi
    class_names = cfg.data.class_names
    saved = []

    print("\n[Visualization] Generating all plots...")

    # 1. Training curves
    saved.append(plot_training_curves(history, output_dir=output_dir, dpi=dpi))

    # 2. Confusion matrix
    if "confusion_matrix" in test_metrics:
        saved.append(plot_confusion_matrix(
            np.array(test_metrics["confusion_matrix"]),
            class_names=class_names,
            output_dir=output_dir,
            title=f"Confusion Matrix — Test Set (Acc={test_metrics['accuracy']:.3f})",
            dpi=dpi,
        ))

    # 3. Per-class performance
    saved.append(plot_per_class_performance(test_metrics, class_names, output_dir, dpi))

    # 4. Loss distribution
    if per_sample_losses is not None:
        saved.append(plot_loss_distribution(
            per_sample_losses, is_noisy=is_noisy,
            output_dir=output_dir, epoch="final", dpi=dpi,
        ))

    # 5. Noise rate history
    nr_vals = [v for v in history.get("estimated_noise_rate", []) if v is not None]
    if nr_vals:
        saved.append(plot_noise_estimation_history(
            nr_vals,
            true_rate=cfg.data.noise_rate,
            output_dir=output_dir, dpi=dpi,
        ))

    # 6. Embeddings
    if embeddings is not None and true_labels is not None:
        saved.append(plot_embeddings(
            embeddings, true_labels, class_names,
            output_dir=output_dir,
            method="both",
            sample_size=cfg.viz.embedding_sample_size,
            dpi=dpi,
            umap_n_neighbors=cfg.viz.umap_n_neighbors,
            umap_min_dist=cfg.viz.umap_min_dist,
        ))

    print(f"[Visualization] {len(saved)} plots saved to: {output_dir}")
    return saved
