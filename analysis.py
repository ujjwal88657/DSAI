"""
analysis.py
Post-training analysis runner for the noisy-label content moderation project.

Loads the best checkpoint, evaluates requested splits, saves metrics and
predictions, computes train loss/GMM noise diagnostics, and regenerates
publication-ready visualizations.
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier import build_model
from config import CFG
from dataset import DataModule
from helpers import get_device
from metrics import compute_loss, compute_metrics, evaluate_noise_detection, predict
from noise_strategies import GaussianMixtureNoiseSeparator
from plots import (
    plot_confusion_matrix,
    plot_embeddings,
    plot_loss_distribution,
    plot_noise_estimation_history,
    plot_per_class_performance,
    plot_training_curves,
)


SCALAR_METRIC_KEYS = [
    "loss",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
]


def parse_args():
    p = argparse.ArgumentParser(description="Generate metrics and visualizations from a trained checkpoint")
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    p.add_argument("--history_path", type=str, default="./logs/training_log.json")
    p.add_argument("--output_dir", type=str, default="./analysis_outputs")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    p.add_argument("--embedding_split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--skip_embeddings", action="store_true")
    p.add_argument("--skip_loss_analysis", action="store_true")
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--text_column", type=str, default=None)
    p.add_argument("--label_column", type=str, default=None)
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--noise_rate", type=float, default=None)
    p.add_argument("--noise_type", type=str, default=None, choices=["symmetric", "asymmetric", "instance"])
    p.add_argument("--no_noise", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train first with `python main.py`, "
            "or pass --checkpoint with the correct file."
        )

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def apply_overrides(cfg, args, checkpoint: Dict):
    ckpt_cfg = checkpoint.get("config", {})

    model_name = args.model_name or ckpt_cfg.get("model_name")
    if model_name:
        cfg.model.model_name = model_name
        cfg.data.tokenizer_name = model_name

    if args.dataset_path:
        cfg.data.dataset_path = args.dataset_path
    if args.text_column:
        cfg.data.text_column = args.text_column
    if args.label_column:
        cfg.data.label_column = args.label_column
    if args.max_seq_len:
        cfg.data.max_seq_len = args.max_seq_len
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.noise_type:
        cfg.data.noise_type = args.noise_type
    elif "noise_type" in ckpt_cfg:
        cfg.data.noise_type = ckpt_cfg["noise_type"]

    if args.noise_rate is not None:
        cfg.data.noise_rate = args.noise_rate
    elif "noise_rate" in ckpt_cfg:
        cfg.data.noise_rate = float(ckpt_cfg["noise_rate"])

    if args.no_noise or cfg.data.noise_rate <= 0:
        cfg.data.simulate_noise = False
        cfg.data.noise_rate = 0.0

    if "num_classes" in ckpt_cfg:
        cfg.data.num_classes = int(ckpt_cfg["num_classes"])
        cfg.model.num_classes = int(ckpt_cfg["num_classes"])

    cfg.training.device = args.device
    cfg.viz.output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)


def load_history(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        print(f"[Analysis] Training history not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_split_loader_and_df(dm: DataModule, split: str):
    if split == "train":
        return dm.get_train_loader(shuffle=False), dm.train_df
    if split == "val":
        return dm.get_val_loader(), dm.val_df
    if split == "test":
        return dm.get_test_loader(), dm.test_df
    raise ValueError(f"Unknown split: {split}")


def class_name(class_names: List[str], label: int) -> str:
    return class_names[label] if 0 <= int(label) < len(class_names) else str(label)


def save_predictions(
    output_dir: str,
    split: str,
    df: pd.DataFrame,
    preds: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
) -> str:
    out = pd.DataFrame({
        "text": df["text"].tolist(),
        "true_label": labels.astype(int),
        "pred_label": preds.astype(int),
        "true_class": [class_name(class_names, x) for x in labels],
        "pred_class": [class_name(class_names, x) for x in preds],
        "is_correct": (preds == labels).astype(int),
    })
    for idx, name in enumerate(class_names):
        out[f"prob_{name}"] = probs[:, idx]

    path = os.path.join(output_dir, f"{split}_predictions.csv")
    out.to_csv(path, index=False, encoding="utf-8")
    return path


def evaluate_split(
    model,
    loader,
    df: pd.DataFrame,
    split: str,
    cfg,
    device: torch.device,
    return_embeddings: bool,
    output_dir: str,
) -> Tuple[Dict, Optional[np.ndarray], str]:
    preds, probs, labels, embeddings = predict(
        model, loader, device, return_embeddings=return_embeddings
    )
    metrics = compute_metrics(preds, labels, class_names=cfg.data.class_names)
    metrics["loss"] = compute_loss(probs, labels)
    predictions_path = save_predictions(
        output_dir, split, df, preds, probs, labels, cfg.data.class_names
    )
    metrics["num_samples"] = int(len(labels))
    return metrics, embeddings, predictions_path


def collect_train_loss_diagnostics(model, loader, device: torch.device) -> Dict:
    model.eval()
    losses, labels, original_labels, noisy_flags, indices = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            batch_labels = batch["label"].to(device)

            out = model(ids, mask, token_type_ids)
            batch_losses = F.cross_entropy(out["logits"], batch_labels, reduction="none")

            losses.extend(batch_losses.cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())
            original_labels.extend(batch["original_label"].cpu().numpy().tolist())
            noisy_flags.extend(batch["is_noisy"].cpu().numpy().tolist())
            indices.extend(batch["index"].cpu().numpy().tolist())

    return {
        "loss": np.array(losses, dtype=float),
        "label": np.array(labels, dtype=int),
        "original_label": np.array(original_labels, dtype=int),
        "is_noisy": np.array(noisy_flags, dtype=int),
        "index": np.array(indices, dtype=int),
    }


def save_metric_tables(output_dir: str, metrics_by_split: Dict[str, Dict]) -> Dict[str, str]:
    summary_rows = []
    per_class_rows = []
    report_lines = []

    for split, metrics in metrics_by_split.items():
        row = {"split": split, "num_samples": metrics.get("num_samples")}
        for key in SCALAR_METRIC_KEYS:
            row[key] = metrics.get(key)
        summary_rows.append(row)

        report_lines.append(f"===== {split.upper()} =====")
        report_lines.append(metrics.get("classification_report", ""))
        report_lines.append("")

        class_names = metrics.get("class_names", [])
        for idx, name in enumerate(class_names):
            per_class_rows.append({
                "split": split,
                "class_index": idx,
                "class_name": name,
                "precision": metrics.get("per_class_precision", [None])[idx],
                "recall": metrics.get("per_class_recall", [None])[idx],
                "f1": metrics.get("per_class_f1", [None])[idx],
            })

    paths = {
        "summary_csv": os.path.join(output_dir, "metrics_summary.csv"),
        "per_class_csv": os.path.join(output_dir, "per_class_metrics.csv"),
        "reports_txt": os.path.join(output_dir, "classification_reports.txt"),
    }
    pd.DataFrame(summary_rows).to_csv(paths["summary_csv"], index=False)
    pd.DataFrame(per_class_rows).to_csv(paths["per_class_csv"], index=False)
    with open(paths["reports_txt"], "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    return paths


def main():
    args = parse_args()
    set_seed(args.seed)

    checkpoint = load_checkpoint(args.checkpoint)
    cfg = CFG
    apply_overrides(cfg, args, checkpoint)

    device = get_device(cfg.training.device, strict=False)
    print(f"[Analysis] Device: {device}")

    print("[Analysis] Loading data...")
    dm = DataModule(cfg)
    dm.setup()

    print("[Analysis] Building model and loading checkpoint...")
    model = build_model(cfg, device)
    model.load_state_dict(checkpoint["model1_state"])
    model.eval()

    metrics_by_split = {}
    embeddings_by_split = {}
    prediction_paths = {}

    for split in args.splits:
        print(f"\n[Analysis] Evaluating {split} split...")
        loader, df = get_split_loader_and_df(dm, split)
        return_embeddings = (split == args.embedding_split and not args.skip_embeddings)
        split_metrics, embeddings, pred_path = evaluate_split(
            model=model,
            loader=loader,
            df=df,
            split=split,
            cfg=cfg,
            device=device,
            return_embeddings=return_embeddings,
            output_dir=args.output_dir,
        )
        split_metrics["class_names"] = cfg.data.class_names
        metrics_by_split[split] = split_metrics
        embeddings_by_split[split] = embeddings
        prediction_paths[split] = pred_path

        print(f"  Accuracy: {split_metrics['accuracy']:.4f}")
        print(f"  F1 weighted: {split_metrics['f1_weighted']:.4f}")

    table_paths = save_metric_tables(args.output_dir, metrics_by_split)
    metrics_json_path = os.path.join(args.output_dir, "metrics_all_splits.json")
    write_json(metrics_json_path, metrics_by_split)

    history = load_history(args.history_path)
    saved_plots = []

    if history and history.get("train_loss"):
        saved_plots.append(plot_training_curves(history, output_dir=args.output_dir, dpi=cfg.viz.dpi))

    plot_split = "test" if "test" in metrics_by_split else args.splits[0]
    plot_metrics = metrics_by_split[plot_split]
    saved_plots.append(plot_confusion_matrix(
        np.array(plot_metrics["confusion_matrix"]),
        class_names=cfg.data.class_names,
        output_dir=args.output_dir,
        title=f"Confusion Matrix - {plot_split.title()} Set (Acc={plot_metrics['accuracy']:.3f})",
        dpi=cfg.viz.dpi,
    ))
    saved_plots.append(plot_per_class_performance(
        plot_metrics, cfg.data.class_names, output_dir=args.output_dir, dpi=cfg.viz.dpi
    ))

    loss_diagnostics_path = None
    gmm_stats = {}
    noise_detection_metrics = {}

    if not args.skip_loss_analysis:
        print("\n[Analysis] Computing train loss distribution and GMM diagnostics...")
        train_loader, _ = get_split_loader_and_df(dm, "train")
        loss_diag = collect_train_loss_diagnostics(model, train_loader, device)

        gmm = GaussianMixtureNoiseSeparator(p_threshold=cfg.training.p_threshold)
        p_clean, is_clean, est_noise = gmm.fit_predict(loss_diag["loss"])
        gmm_stats = gmm.get_stats()
        gmm_stats.update({
            "estimated_noise_rate": float(est_noise),
            "num_predicted_clean": int(is_clean.sum()),
            "num_predicted_noisy": int((~is_clean).sum()),
        })

        if len(np.unique(loss_diag["is_noisy"])) > 1:
            noise_detection_metrics = evaluate_noise_detection(p_clean, loss_diag["is_noisy"])
            noise_flags_for_plot = loss_diag["is_noisy"]
        else:
            noise_flags_for_plot = None

        loss_table = pd.DataFrame({
            "index": loss_diag["index"],
            "label": loss_diag["label"],
            "original_label": loss_diag["original_label"],
            "is_noisy": loss_diag["is_noisy"],
            "loss": loss_diag["loss"],
            "p_clean": p_clean,
            "predicted_clean": is_clean.astype(int),
        })
        loss_diagnostics_path = os.path.join(args.output_dir, "train_loss_diagnostics.csv")
        loss_table.to_csv(loss_diagnostics_path, index=False)

        saved_plots.append(plot_loss_distribution(
            loss_diag["loss"],
            is_noisy=noise_flags_for_plot,
            output_dir=args.output_dir,
            epoch="analysis",
            gmm_params=gmm_stats,
            dpi=cfg.viz.dpi,
        ))

    if history:
        nr_vals = [v for v in history.get("estimated_noise_rate", []) if v is not None]
        if nr_vals:
            saved_plots.append(plot_noise_estimation_history(
                nr_vals,
                true_rate=cfg.data.noise_rate,
                output_dir=args.output_dir,
                dpi=cfg.viz.dpi,
            ))

    emb = embeddings_by_split.get(args.embedding_split)
    if emb is not None:
        _, emb_df = get_split_loader_and_df(dm, args.embedding_split)
        saved_plots.append(plot_embeddings(
            emb,
            emb_df["label"].values,
            cfg.data.class_names,
            output_dir=args.output_dir,
            method="both",
            sample_size=cfg.viz.embedding_sample_size,
            dpi=cfg.viz.dpi,
            umap_n_neighbors=cfg.viz.umap_n_neighbors,
            umap_min_dist=cfg.viz.umap_min_dist,
        ))

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "output_dir": os.path.abspath(args.output_dir),
        "metrics_json": metrics_json_path,
        "tables": table_paths,
        "prediction_files": prediction_paths,
        "loss_diagnostics_csv": loss_diagnostics_path,
        "gmm_stats": gmm_stats,
        "noise_detection_metrics": noise_detection_metrics,
        "plots": saved_plots,
    }
    write_json(os.path.join(args.output_dir, "analysis_report.json"), report)

    print("\n[Analysis] Complete.")
    print(f"  Metrics: {metrics_json_path}")
    print(f"  Summary CSV: {table_paths['summary_csv']}")
    print(f"  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
