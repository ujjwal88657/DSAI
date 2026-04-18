"""
main.py
Entry point for the Social Media Content Moderation project.
Orchestrates the full pipeline:
  data → model → train (co-teaching + small-loss + DivideMix + bootstrapping) → evaluate → visualize
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch

# ---- Add project root to path ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CFG, Config, DataConfig, ModelConfig, TrainingConfig, VisualizationConfig
from dataset import DataModule
from classifier import build_dual_models
from trainer import Trainer
from metrics import predict
from plots import run_all_visualizations
from noise_strategies import GaussianMixtureNoiseSeparator


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Device selection
# ============================================================

def get_device(cfg) -> torch.device:
    if cfg.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif cfg.training.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Noisy Label Content Moderation")
    p.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    p.add_argument("--noise_rate", type=float, default=None, help="Override noise_rate (0.0-1.0)")
    p.add_argument("--noise_type", type=str, default=None,
                   choices=["symmetric", "asymmetric", "instance"])
    p.add_argument("--dataset_path", type=str, default=None, help="CSV dataset path")
    p.add_argument("--text_column", type=str, default=None, help="Text column in the CSV")
    p.add_argument("--label_column", type=str, default=None, help="Label column in the CSV")
    p.add_argument("--loss", type=str, default=None,
                   choices=["ce", "sce", "gce", "mae"], help="Loss function")
    p.add_argument("--lr", type=float, default=None, help="Learning rate")
    p.add_argument("--model_name", type=str, default=None, help="HuggingFace model name/path")
    p.add_argument("--max_seq_len", type=int, default=None, help="Maximum tokenizer sequence length")
    p.add_argument("--no_noise", action="store_true", help="Disable synthetic noise injection")
    p.add_argument("--no_co_teach", action="store_true", help="Disable Co-Teaching")
    p.add_argument("--no_divide_mix", action="store_true", help="Disable DivideMix GMM")
    p.add_argument("--no_bootstrap", action="store_true", help="Disable bootstrapping")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./visualizations")
    p.add_argument("--fast", action="store_true",
                   help="Fast mode: fewer epochs, smaller model for CI/testing")
    return p.parse_args()


# ============================================================
# Main pipeline
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    # ---- Apply argument overrides ----
    cfg = CFG
    if args.epochs: cfg.training.num_epochs = args.epochs
    if args.batch_size: cfg.training.batch_size = args.batch_size
    if args.dataset_path: cfg.data.dataset_path = args.dataset_path
    if args.text_column: cfg.data.text_column = args.text_column
    if args.label_column: cfg.data.label_column = args.label_column
    if args.noise_rate is not None: cfg.data.noise_rate = args.noise_rate
    if args.noise_type: cfg.data.noise_type = args.noise_type
    if args.loss: cfg.training.loss_type = args.loss
    if args.lr: cfg.training.learning_rate = args.lr
    if args.model_name:
        cfg.model.model_name = args.model_name
        cfg.data.tokenizer_name = args.model_name
    if args.max_seq_len: cfg.data.max_seq_len = args.max_seq_len
    if args.no_noise:
        cfg.data.simulate_noise = False
        cfg.data.noise_rate = 0.0
    if args.no_co_teach: cfg.training.use_co_teaching = False
    if args.no_divide_mix: cfg.training.use_divide_mix = False
    if args.no_bootstrap: cfg.training.use_bootstrapping = False
    if args.output_dir: cfg.viz.output_dir = args.output_dir

    # Fast mode for quick testing
    if args.fast:
        cfg.training.num_epochs = 3
        cfg.training.batch_size = 16
        cfg.data.num_classes = 3
        cfg.training.log_every_n_steps = 5
        print("[Fast Mode] Reduced epochs and batch size for testing.")

    cfg.display()

    # ---- Device ----
    device = get_device(cfg)

    # ============================================================
    # STEP 1: Data
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)

    dm = DataModule(cfg)
    dm.setup()

    # ============================================================
    # STEP 2: Models
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: MODEL INITIALIZATION")
    print("=" * 60)

    model1, model2 = build_dual_models(cfg, device)

    # ============================================================
    # STEP 3: Training
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING")
    print("=" * 60)
    print(f"  Loss: {cfg.training.loss_type.upper()}")
    print(f"  Co-Teaching: {cfg.training.use_co_teaching}")
    print(f"  DivideMix: {cfg.training.use_divide_mix}")
    print(f"  Bootstrapping: {cfg.training.use_bootstrapping}")
    noise_status = (
        f"{cfg.data.noise_type} @ {cfg.data.noise_rate:.0%}"
        if cfg.data.simulate_noise else "disabled"
    )
    print(f"  Synthetic noise: {noise_status}")

    trainer = Trainer(cfg, model1, model2, dm, device)
    history = trainer.train()

    # ============================================================
    # STEP 4: Load Best Model & Evaluate
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: EVALUATION")
    print("=" * 60)

    trainer.load_best_checkpoint()
    test_loader = dm.get_test_loader()

    # Full evaluation with predictions and embeddings
    test_preds, test_probs, test_labels, test_embeddings = predict(
        model1, test_loader, device, return_embeddings=True
    )

    from metrics import compute_metrics, compute_loss
    test_metrics = compute_metrics(test_preds, test_labels, class_names=cfg.data.class_names)
    test_metrics["loss"] = compute_loss(test_probs, test_labels)

    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"  F1 Macro:        {test_metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted:     {test_metrics['f1_weighted']:.4f}")
    print(f"  Precision Macro: {test_metrics['precision_macro']:.4f}")
    print(f"  Recall Macro:    {test_metrics['recall_macro']:.4f}")
    print(f"\n{test_metrics['classification_report']}")

    # Save test results to JSON
    results_path = os.path.join(cfg.training.log_dir, "test_results.json")
    save_metrics = {k: v for k, v in test_metrics.items() if k != "classification_report"}
    with open(results_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"[Results] Saved to {results_path}")

    # ============================================================
    # STEP 5: GMM on test losses for visualization
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 5: LOSS ANALYSIS")
    print("=" * 60)

    # Compute per-sample losses on training data for DivideMix visualization
    train_loader_eval = dm.get_train_loader(shuffle=False)
    import torch.nn.functional as F_torch

    model1.eval()
    all_losses, all_noisy_flags = [], []
    with torch.no_grad():
        for batch in train_loader_eval:
            ids = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            tt = batch["token_type_ids"].to(device)
            lbl = batch["label"].to(device)
            out = model1(ids, msk, tt)
            losses = F_torch.cross_entropy(out["logits"], lbl, reduction="none")
            all_losses.extend(losses.cpu().numpy().tolist())
            all_noisy_flags.extend(batch.get("is_noisy", torch.zeros(len(lbl))).numpy().tolist())

    per_sample_losses = np.array(all_losses)
    is_noisy_arr = np.array(all_noisy_flags)

    # Fit GMM for final report
    gmm = GaussianMixtureNoiseSeparator(p_threshold=cfg.training.p_threshold)
    p_clean, is_clean, est_noise = gmm.fit_predict(per_sample_losses)
    print(f"  Final GMM estimated noise rate: {est_noise:.2%}")
    print(f"  True noise rate: {cfg.data.noise_rate:.2%}")

    # ============================================================
    # STEP 6: Visualizations
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 6: VISUALIZATIONS")
    print("=" * 60)

    run_all_visualizations(
        history=history,
        test_metrics=test_metrics,
        per_sample_losses=per_sample_losses,
        is_noisy=is_noisy_arr,
        embeddings=test_embeddings,
        true_labels=test_labels,
        cfg=cfg,
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"  Best Val F1: {trainer.best_val_f1:.4f} (epoch {trainer.best_epoch})")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 Weighted: {test_metrics['f1_weighted']:.4f}")
    print(f"  Visualizations: {cfg.viz.output_dir}/")
    print(f"  Checkpoints:    {cfg.model.checkpoint_dir}/")
    print(f"  Logs:           {cfg.training.log_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
