"""
demo.py
Standalone demo that runs the COMPLETE pipeline using a lightweight
TF-IDF + MLP backbone instead of BERT (no model download required).
Demonstrates all noise-handling strategies with identical API.

For the full BERT version: python main.py --fast
"""

import os, sys, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Inline imports from project ──────────────────────────────────────────────
from dataset import generate_hinglish_dataset, apply_noise
from robust_losses import (
    SymmetricCrossEntropyLoss, CrossEntropyLoss,
    BootstrappingLoss, compute_per_sample_loss
)
from noise_strategies import (
    SmallLossTrick, CoTeaching, GaussianMixtureNoiseSeparator,
    LabelRefurbishmentStore, NoiseRateEstimator
)
from metrics import compute_metrics, compute_loss
from plots import (
    plot_training_curves, plot_confusion_matrix,
    plot_loss_distribution, plot_per_class_performance
)
from helpers import get_device


# ─────────────────────────────────────────────────────────────────────────────
# Config (lightweight)
# ─────────────────────────────────────────────────────────────────────────────
class DemoCfg:
    num_classes = 3
    class_names = ["hate", "offensive", "neutral"]
    noise_rate   = 0.30
    noise_type   = "asymmetric"
    num_epochs   = 12
    batch_size   = 64
    lr           = 3e-3
    forget_rate  = 0.20
    num_gradual  = 5
    sce_alpha    = 0.1
    sce_beta     = 1.0
    bootstrap_beta  = 0.8
    bootstrap_start = 5
    p_threshold  = 0.5
    keep_ratio_initial = 1.0
    keep_ratio_final   = 0.70
    small_loss_start   = 4
    viz_dir   = "./visualizations"
    log_dir   = "./logs"
    seed      = 42

CFG = DemoCfg()
os.makedirs(CFG.viz_dir, exist_ok=True)
os.makedirs(CFG.log_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG.seed)


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF Dataset
# ─────────────────────────────────────────────────────────────────────────────
class TFIDFDataset(Dataset):
    def __init__(self, X, labels, orig_labels=None, is_noisy=None, indices=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.orig   = torch.tensor(orig_labels if orig_labels is not None else labels, dtype=torch.long)
        self.noisy  = torch.tensor(is_noisy if is_noisy is not None else [0]*len(labels), dtype=torch.long)
        self.idx    = torch.tensor(indices if indices is not None else list(range(len(labels))), dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return {
            "features": self.X[i],
            "label": self.labels[i],
            "original_label": self.orig[i],
            "is_noisy": self.noisy[i],
            "index": self.idx[i],
        }

    def update_labels(self, new_labels):
        self.labels = torch.tensor(new_labels, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# MLP Classifier (BERT substitute for demo)
# ─────────────────────────────────────────────────────────────────────────────
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, features, return_embeddings=False):
        # Extract penultimate embedding
        h = features
        for i, layer in enumerate(self.net):
            h = layer(h)
            if i == 9:   # after 3rd block, before final linear
                emb = h
        logits = h
        if return_embeddings:
            return {"logits": logits, "embeddings": emb}
        return {"logits": logits}


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────
def prepare_data():
    print("[Data] Generating Hinglish dataset...")
    df = generate_hinglish_dataset(n_samples=3000, seed=CFG.seed)
    df = apply_noise(df, CFG.noise_type, CFG.noise_rate, CFG.num_classes, CFG.seed)

    # Split
    train_val, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=CFG.seed)
    train_df, val_df   = train_test_split(train_val, test_size=0.15/0.85, stratify=train_val["label"], random_state=CFG.seed)

    # Restore clean labels for val/test
    val_df  = val_df.copy();  val_df["label"]  = val_df["original_label"]
    test_df = test_df.copy(); test_df["label"] = test_df["original_label"]

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"  Train noise: {train_df['is_noisy'].mean():.2%}")

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["text"]).toarray()
    X_val   = vectorizer.transform(val_df["text"]).toarray()
    X_test  = vectorizer.transform(test_df["text"]).toarray()

    def make_ds(df, X, idx_offset=0):
        noisy_col = df["is_noisy"].values if "is_noisy" in df.columns else None
        return TFIDFDataset(
            X, df["label"].values,
            orig_labels=df["original_label"].values if "original_label" in df.columns else df["label"].values,
            is_noisy=noisy_col,
            indices=list(range(len(df))),
        )

    train_ds = make_ds(train_df, X_train)
    val_ds   = make_ds(val_df, X_val)
    test_ds  = make_ds(test_df, X_test)

    return train_ds, val_ds, test_ds, X_train.shape[1], train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def run_eval(model, loader, device):
    model.eval()
    preds, probs_all, labels_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            out   = model(feats)
            p     = F.softmax(out["logits"], dim=1)
            preds.extend(p.argmax(1).cpu().numpy())
            probs_all.extend(p.cpu().numpy())
            labels_all.extend(batch["label"].numpy())
    return np.array(preds), np.array(probs_all), np.array(labels_all)


def get_per_sample_losses(model, loader, device):
    model.eval()
    losses, noisy_flags, all_idx = [], [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            lbl   = batch["label"].to(device)
            out   = model(feats)
            l     = F.cross_entropy(out["logits"], lbl, reduction="none")
            losses.extend(l.cpu().numpy())
            noisy_flags.extend(batch["is_noisy"].numpy())
            all_idx.extend(batch["index"].numpy())
    return np.array(losses), np.array(noisy_flags), np.array(all_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Full Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train():
    device = get_device("auto")
    print(f"[Device] {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, input_dim, train_df, test_df = prepare_data()

    loader_kwargs = dict(batch_size=CFG.batch_size, num_workers=0, pin_memory=False)
    loader1 = DataLoader(train_ds, shuffle=True,  drop_last=True, **loader_kwargs)
    loader2 = DataLoader(train_ds, shuffle=True,  drop_last=True, **loader_kwargs)
    val_loader  = DataLoader(val_ds,  shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    eval_train_loader = DataLoader(train_ds, shuffle=False, **loader_kwargs)

    # ── Models ───────────────────────────────────────────────────────────────
    model1 = MLPClassifier(input_dim, CFG.num_classes).to(device)
    model2 = MLPClassifier(input_dim, CFG.num_classes).to(device)

    # ── Losses ───────────────────────────────────────────────────────────────
    sce_loss = SymmetricCrossEntropyLoss(CFG.num_classes, CFG.sce_alpha, CFG.sce_beta, reduction="none")
    boot_loss = BootstrappingLoss(CFG.num_classes, CFG.bootstrap_beta, reduction="none")

    # ── Optimizers ───────────────────────────────────────────────────────────
    opt1 = torch.optim.AdamW(model1.parameters(), lr=CFG.lr, weight_decay=1e-4)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=CFG.lr, weight_decay=1e-4)

    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=CFG.num_epochs)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=CFG.num_epochs)

    # ── Noise strategies ──────────────────────────────────────────────────────
    co_teach = CoTeaching(CFG.forget_rate, CFG.num_gradual, CFG.num_epochs)
    small_loss = SmallLossTrick(CFG.keep_ratio_initial, CFG.keep_ratio_final,
                                CFG.small_loss_start, CFG.num_epochs)
    refurb = LabelRefurbishmentStore(len(train_ds), CFG.num_classes, alpha=0.9)
    noise_est = NoiseRateEstimator()

    # ── History ───────────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_loss_m2": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "estimated_noise_rate": [], "keep_ratio": [], "forget_rate": [], "epoch_time": [],
    }

    best_val_f1, best_state = 0.0, None

    print("\n[Training] Starting co-teaching loop...\n")
    for epoch in range(CFG.num_epochs):
        t0 = time.time()
        model1.train(); model2.train()
        epoch_l1, epoch_l2, steps = 0.0, 0.0, 0

        for batch1, batch2 in zip(loader1, loader2):
            f1 = batch1["features"].to(device); l1 = batch1["label"].to(device)
            f2 = batch2["features"].to(device); l2 = batch2["label"].to(device)

            # ── Co-Teaching selection ──────────────────────────────────────
            with torch.no_grad():
                lo1 = model1(f1)["logits"]; lo2 = model2(f2)["logits"]
            ps1 = compute_per_sample_loss(lo1, l1)
            ps2 = compute_per_sample_loss(lo2, l2)

            ci1 = co_teach.select_indices(ps1, epoch)   # model1 selects from batch1
            ci2 = co_teach.select_indices(ps2, epoch)   # model2 selects from batch2

            # ── Model 1 trains on batch2's clean ──────────────────────────
            opt1.zero_grad()
            out1 = model1(f2[ci2])
            soft1 = F.softmax(out1["logits"], dim=1)
            refurb.update(batch2["index"][ci2.cpu()], soft1)

            if epoch >= CFG.bootstrap_start:
                loss1_arr = boot_loss(out1["logits"], l2[ci2])
            else:
                loss1_arr = sce_loss(out1["logits"], l2[ci2])
            loss1 = loss1_arr.mean()
            loss1.backward(); nn.utils.clip_grad_norm_(model1.parameters(), 1.0); opt1.step()

            # ── Model 2 trains on batch1's clean ──────────────────────────
            opt2.zero_grad()
            out2 = model2(f1[ci1])
            if epoch >= CFG.bootstrap_start:
                loss2_arr = boot_loss(out2["logits"], l1[ci1])
            else:
                loss2_arr = sce_loss(out2["logits"], l1[ci1])
            loss2 = loss2_arr.mean()
            loss2.backward(); nn.utils.clip_grad_norm_(model2.parameters(), 1.0); opt2.step()

            epoch_l1 += loss1.item(); epoch_l2 += loss2.item(); steps += 1

        sched1.step(); sched2.step()

        # ── GMM noise estimation ───────────────────────────────────────────
        losses_ep, noisy_ep, _ = get_per_sample_losses(model1, eval_train_loader, device)
        gmm_stats, p_clean, is_clean = noise_est.estimate(losses_ep)
        est_nr = gmm_stats.get("estimated_noise_rate", 0)

        # ── Validation ────────────────────────────────────────────────────
        preds_v, probs_v, labels_v = run_eval(model1, val_loader, device)
        val_metrics = compute_metrics(preds_v, labels_v, CFG.class_names)
        val_loss = compute_loss(probs_v, labels_v)

        # ── Log ───────────────────────────────────────────────────────────
        val_acc = val_metrics["accuracy"]
        val_f1  = val_metrics["f1_weighted"]
        kr      = small_loss.get_keep_ratio(epoch)
        fr      = co_teach.get_forget_rate(epoch)
        et      = time.time() - t0

        history["train_loss"].append(epoch_l1 / steps)
        history["train_loss_m2"].append(epoch_l2 / steps)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["estimated_noise_rate"].append(est_nr)
        history["keep_ratio"].append(kr)
        history["forget_rate"].append(fr)
        history["epoch_time"].append(et)

        print(f"Epoch {epoch+1:02d}/{CFG.num_epochs} | "
              f"L1={epoch_l1/steps:.4f} L2={epoch_l2/steps:.4f} | "
              f"Val Acc={val_acc:.4f} F1={val_f1:.4f} | "
              f"NoiseEst={est_nr:.2%} | FR={fr:.2f} | t={et:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model1.state_dict().items()}
            print(f"  *** Best model saved (F1={val_f1:.4f}) ***")

    # ── Restore best & evaluate on test ──────────────────────────────────────
    model1.load_state_dict(best_state)
    preds_t, probs_t, labels_t = run_eval(model1, test_loader, device)
    test_metrics = compute_metrics(preds_t, labels_t, CFG.class_names)
    test_metrics["loss"] = compute_loss(probs_t, labels_t)

    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(test_metrics["classification_report"])
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  F1 Macro:    {test_metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {test_metrics['f1_weighted']:.4f}")

    # ── Final per-sample losses ───────────────────────────────────────────────
    final_losses, final_noisy, _ = get_per_sample_losses(model1, eval_train_loader, device)
    final_gmm = GaussianMixtureNoiseSeparator(CFG.p_threshold)
    _, _, final_est_nr = final_gmm.fit_predict(final_losses)
    print(f"\n  Final GMM estimated noise rate: {final_est_nr:.2%} (true: {CFG.noise_rate:.2%})")

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\n[Visualization] Generating plots...")

    plot_training_curves(history, output_dir=CFG.viz_dir)

    plot_confusion_matrix(
        np.array(test_metrics["confusion_matrix"]),
        class_names=CFG.class_names,
        output_dir=CFG.viz_dir,
        title=f"Confusion Matrix — Test (Acc={test_metrics['accuracy']:.3f})",
    )

    plot_loss_distribution(
        final_losses,
        is_noisy=final_noisy,
        output_dir=CFG.viz_dir,
        epoch="final",
        gmm_params=final_gmm.get_stats(),
    )

    plot_per_class_performance(test_metrics, CFG.class_names, CFG.viz_dir)

    # Embeddings (PCA only — UMAP optional)
    model1.eval()
    embs, tl_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model1(batch["features"].to(device), return_embeddings=True)
            if "embeddings" in out:
                embs.append(out["embeddings"].cpu().numpy())
            tl_all.extend(batch["label"].numpy())
    if embs:
        from plots import plot_embeddings
        emb_arr = np.concatenate(embs)
        plot_embeddings(emb_arr, np.array(tl_all), CFG.class_names,
                        output_dir=CFG.viz_dir, method="both", sample_size=500)

    # Save results
    save_me = {k: v for k, v in test_metrics.items() if k != "classification_report"}
    with open(os.path.join(CFG.log_dir, "demo_results.json"), "w") as f:
        json.dump(save_me, f, indent=2)
    with open(os.path.join(CFG.log_dir, "training_log.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"  Best Val F1:       {best_val_f1:.4f}")
    print(f"  Test Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 Weighted:  {test_metrics['f1_weighted']:.4f}")
    print(f"  Plots:             {CFG.viz_dir}/")
    print(f"  Logs:              {CFG.log_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
