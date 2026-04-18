"""
evaluation/metrics.py
Full evaluation suite:
  - Accuracy, Precision, Recall, F1
  - Per-class breakdown
  - Confusion matrix computation
  - Label-level noise detection metrics (if ground truth available)
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


# -----------------------------------------------------------------------
# Core inference loop
# -----------------------------------------------------------------------

def predict(
    model,
    loader,
    device: torch.device,
    return_embeddings: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Run inference over a DataLoader.

    Returns:
      preds       : (N,) int array of predicted class indices
      probs       : (N, C) float array of softmax probabilities
      true_labels : (N,) int array of true labels
      embeddings  : (N, H) float array (only if return_embeddings=True)
    """
    model.eval()
    all_preds, all_probs, all_labels, all_embeddings = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Inference", ncols=80, leave=False):
            ids = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            tt = batch["token_type_ids"].to(device)
            lbl = batch["label"]

            out = model(ids, msk, tt, return_embeddings=return_embeddings)
            logits = out["logits"]
            probs = F.softmax(logits, dim=1)

            all_preds.append(probs.argmax(dim=1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(lbl.numpy())

            if return_embeddings and "embeddings" in out:
                all_embeddings.append(out["embeddings"].cpu().numpy())

    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    embeddings = np.concatenate(all_embeddings) if all_embeddings else None

    return preds, probs, labels, embeddings


# -----------------------------------------------------------------------
# Metric computation
# -----------------------------------------------------------------------

def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute full classification metrics."""
    label_ids = np.arange(len(class_names)) if class_names else None
    acc = accuracy_score(labels, preds)

    # Handle zero_division for imbalanced classes
    precision_macro = precision_score(labels, preds, labels=label_ids, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, labels=label_ids, average="macro", zero_division=0)
    f1_macro = f1_score(labels, preds, labels=label_ids, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, labels=label_ids, average="weighted", zero_division=0)

    per_class_f1 = f1_score(labels, preds, labels=label_ids, average=None, zero_division=0)
    per_class_precision = precision_score(labels, preds, labels=label_ids, average=None, zero_division=0)
    per_class_recall = recall_score(labels, preds, labels=label_ids, average=None, zero_division=0)

    cm = confusion_matrix(labels, preds, labels=label_ids)

    report = classification_report(
        labels, preds,
        labels=label_ids,
        target_names=class_names,
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def compute_loss(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute cross-entropy loss from probability arrays."""
    eps = 1e-7
    clipped = np.clip(probs, eps, 1.0 - eps)
    ce = -np.log(clipped[np.arange(len(labels)), labels])
    return float(ce.mean())


# -----------------------------------------------------------------------
# High-level evaluator
# -----------------------------------------------------------------------

def evaluate_model(
    model,
    loader,
    device: torch.device,
    cfg,
    split: str = "test",
    return_predictions: bool = False,
) -> Dict:
    """
    Full evaluation pipeline.
    Returns metrics dict; optionally also predictions and probs.
    """
    class_names = cfg.data.class_names

    preds, probs, labels, _ = predict(model, loader, device, return_embeddings=False)

    loss = compute_loss(probs, labels)
    metrics = compute_metrics(preds, labels, class_names=class_names)
    metrics["loss"] = loss

    print(f"\n  [{split.upper()} EVALUATION]")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"\n  Per-class F1: {dict(zip(class_names, [f'{v:.3f}' for v in metrics['per_class_f1']]))}")
    print(f"\n  Classification Report:\n{metrics['classification_report']}")

    if return_predictions:
        metrics["preds"] = preds
        metrics["probs"] = probs
        metrics["labels"] = labels

    return metrics


# -----------------------------------------------------------------------
# Noise detection evaluation (oracle metric)
# -----------------------------------------------------------------------

def evaluate_noise_detection(
    per_sample_probs_clean: np.ndarray,
    true_noise_mask: np.ndarray,
) -> Dict:
    """
    Evaluate how well the GMM clean probability scores separate
    true-labeled from false-labeled examples.

    per_sample_probs_clean: probability each sample is clean (from GMM)
    true_noise_mask: ground truth binary mask (1=noisy, 0=clean)
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    # p_clean = 1 means "model thinks it's clean"
    # true_noise_mask = 0 means it IS clean
    # So AUROC should be computed for: score=p_clean, label=(1-noise_mask)
    true_clean = 1 - true_noise_mask.astype(int)

    auroc = roc_auc_score(true_clean, per_sample_probs_clean)
    avg_prec = average_precision_score(true_clean, per_sample_probs_clean)

    # Threshold at 0.5
    predicted_clean = (per_sample_probs_clean > 0.5).astype(int)
    precision = precision_score(true_clean, predicted_clean, zero_division=0)
    recall = recall_score(true_clean, predicted_clean, zero_division=0)
    f1 = f1_score(true_clean, predicted_clean, zero_division=0)

    return {
        "noise_detection_auroc": float(auroc),
        "noise_detection_avg_precision": float(avg_prec),
        "noise_detection_f1": float(f1),
        "noise_detection_precision": float(precision),
        "noise_detection_recall": float(recall),
    }
