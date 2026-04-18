"""
training/trainer.py
Full end-to-end training pipeline integrating:
  - Co-Teaching (dual model)
  - Small-loss trick
  - DivideMix-style GMM label separation
  - Bootstrapping / label correction
  - SCE or other robust loss
  - Logging of all metrics
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from robust_losses import (
    build_loss, BootstrappingLoss, compute_per_sample_loss, SymmetricCrossEntropyLoss
)
from noise_strategies import (
    SmallLossTrick, CoTeaching, GaussianMixtureNoiseSeparator,
    LabelRefurbishmentStore, NoiseRateEstimator
)
from metrics import evaluate_model


class TrainingLogger:
    """Lightweight logger that accumulates metrics and saves to JSON."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "training_log.json")
        self.history: Dict[str, List] = {
            "train_loss": [], "train_loss_m2": [],
            "val_loss": [], "val_acc": [], "val_f1": [],
            "estimated_noise_rate": [],
            "keep_ratio": [], "forget_rate": [],
            "epoch_time": [],
        }

    def log_epoch(self, metrics: Dict):
        for k, v in metrics.items():
            if k in self.history:
                self.history[k].append(float(v) if v is not None else None)

    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)


class Trainer:
    """
    Unified trainer implementing all noise-robust strategies from the survey.

    Training flow per epoch:
    1. Co-Teaching forward/backward on paired batches
    2. Small-loss selection applied inside co-teaching
    3. After small_loss_start_epoch: also apply GMM to epoch-level losses
    4. After bootstrap_start_epoch: label refurbishment via EMA predictions
    5. Evaluate on clean validation set
    6. Save best checkpoint
    """

    def __init__(self, cfg, model1, model2, data_module, device):
        self.cfg = cfg
        self.tcfg = cfg.training
        self.model1 = model1
        self.model2 = model2
        self.data_module = data_module
        self.device = device

        # ---- Loss functions ----
        # Primary per-sample loss (SCE by default)
        self.loss_fn = build_loss(cfg)
        # Bootstrapping loss for label correction phase
        self.bootstrap_loss = BootstrappingLoss(
            num_classes=cfg.data.num_classes,
            beta=self.tcfg.bootstrap_beta,
            reduction="none",
        )

        # ---- Noise strategies ----
        n_train = len(data_module.train_df)
        self.small_loss = SmallLossTrick(
            keep_ratio_initial=self.tcfg.keep_ratio_initial,
            keep_ratio_final=self.tcfg.keep_ratio_final,
            start_epoch=self.tcfg.small_loss_start_epoch,
            total_epochs=self.tcfg.num_epochs,
        )
        self.co_teach = CoTeaching(
            forget_rate=self.tcfg.forget_rate,
            num_gradual=self.tcfg.num_gradual,
            total_epochs=self.tcfg.num_epochs,
            exponent=self.tcfg.exponent,
        )
        self.gmm = GaussianMixtureNoiseSeparator(p_threshold=self.tcfg.p_threshold)
        self.refurb_store = LabelRefurbishmentStore(
            n_samples=n_train,
            num_classes=cfg.data.num_classes,
            alpha=0.9,
            device=str(device),
        )
        self.noise_estimator = NoiseRateEstimator()

        # ---- Optimizers ----
        self.opt1, self.opt2 = self._build_optimizers()

        # ---- Schedulers (created after knowing total steps) ----
        self.sched1 = None
        self.sched2 = None

        # ---- Logger ----
        self.logger = TrainingLogger(self.tcfg.log_dir)
        self.best_val_f1 = 0.0
        self.best_epoch = 0

        # Storage for epoch-level losses (for GMM and visualization)
        self.epoch_losses_m1: List[float] = []     # per-sample losses from model1
        self.epoch_indices: List[int] = []
        self.all_epoch_loss_history: List[np.ndarray] = []  # shape: (epochs, N)

    def _build_optimizers(self) -> Tuple:
        def _opt(model):
            no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
            grouped = [
                {
                    "params": [p for n, p in model.named_parameters()
                               if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.tcfg.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters()
                               if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            return AdamW(grouped, lr=self.tcfg.learning_rate)

        return _opt(self.model1), _opt(self.model2)

    def _build_schedulers(self, steps_per_epoch: int):
        total_steps = self.tcfg.num_epochs * steps_per_epoch
        warmup_steps = int(total_steps * self.tcfg.warmup_ratio)

        def _sched(opt):
            return torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.tcfg.learning_rate,
                total_steps=total_steps,
                pct_start=self.tcfg.warmup_ratio,
                anneal_strategy="cos",
            )

        self.sched1 = _sched(self.opt1)
        self.sched2 = _sched(self.opt2)

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def _train_step_co_teach(
        self,
        batch1: Dict,
        batch2: Dict,
        epoch: int,
    ) -> Tuple[float, float, Dict]:
        """
        One Co-Teaching step with integrated small-loss selection.
        Each model selects clean examples for the OTHER model to learn from.
        """
        def _tensors(batch):
            return (
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                batch["token_type_ids"].to(self.device),
                batch["label"].to(self.device),
                batch["index"],
            )

        ids1, msk1, tt1, lbl1, idx1 = _tensors(batch1)
        ids2, msk2, tt2, lbl2, idx2 = _tensors(batch2)

        # ------ Compute per-sample losses for selection ------
        self.model1.eval(); self.model2.eval()
        with torch.no_grad():
            out1_sel = self.model1(ids1, msk1, tt1)
            out2_sel = self.model2(ids2, msk2, tt2)

        losses_sel1 = compute_per_sample_loss(out1_sel["logits"], lbl1)
        losses_sel2 = compute_per_sample_loss(out2_sel["logits"], lbl2)

        # Store for GMM analysis
        self.epoch_losses_m1.extend(losses_sel1.cpu().numpy().tolist())
        self.epoch_indices.extend(idx1.numpy().tolist())

        # ------ Co-Teaching selection ------
        clean_idx1 = self.co_teach.select_indices(losses_sel1, epoch)  # model1's clean from batch1
        clean_idx2 = self.co_teach.select_indices(losses_sel2, epoch)  # model2's clean from batch2

        # ------ Training ------
        self.model1.train(); self.model2.train()

        # Model 1 trains on batch2's clean examples (selected by model2)
        self.opt1.zero_grad()
        out1_train = self.model1(ids2[clean_idx2], msk2[clean_idx2], tt2[clean_idx2])

        # Apply bootstrapping if past warmup
        if epoch >= self.tcfg.bootstrap_start_epoch and self.tcfg.use_bootstrapping:
            clean_idx2_cpu = clean_idx2.cpu()
            selected_indices = idx2[clean_idx2_cpu] if hasattr(idx2, '__getitem__') else idx2
            soft_preds = F.softmax(out1_train["logits"], dim=1)
            self.refurb_store.update(
                torch.tensor([idx2[i.item()] for i in clean_idx2]),
                soft_preds
            )
            loss1_arr = self.bootstrap_loss(out1_train["logits"], lbl2[clean_idx2])
        else:
            loss1_arr = self.loss_fn(out1_train["logits"], lbl2[clean_idx2])

        loss1 = loss1_arr.mean()
        loss1.backward()
        nn.utils.clip_grad_norm_(self.model1.parameters(), self.tcfg.gradient_clip)
        self.opt1.step()
        if self.sched1: self.sched1.step()

        # Model 2 trains on batch1's clean examples (selected by model1)
        self.opt2.zero_grad()
        out2_train = self.model2(ids1[clean_idx1], msk1[clean_idx1], tt1[clean_idx1])

        if epoch >= self.tcfg.bootstrap_start_epoch and self.tcfg.use_bootstrapping:
            loss2_arr = self.bootstrap_loss(out2_train["logits"], lbl1[clean_idx1])
        else:
            loss2_arr = self.loss_fn(out2_train["logits"], lbl1[clean_idx1])

        loss2 = loss2_arr.mean()
        loss2.backward()
        nn.utils.clip_grad_norm_(self.model2.parameters(), self.tcfg.gradient_clip)
        self.opt2.step()
        if self.sched2: self.sched2.step()

        info = {
            "clean1": len(clean_idx1),
            "clean2": len(clean_idx2),
            "forget_rate": self.co_teach.get_forget_rate(epoch),
        }
        return loss1.item(), loss2.item(), info

    # ------------------------------------------------------------------
    # Epoch-level operations
    # ------------------------------------------------------------------

    def _run_gmm_analysis(self, epoch: int) -> Dict:
        """Fit GMM to accumulated losses from this epoch."""
        losses_arr = np.array(self.epoch_losses_m1)
        stats, p_clean, is_clean = self.noise_estimator.estimate(losses_arr)
        return stats

    def _anneal_bootstrapping(self, epoch: int):
        """Gradually decrease model trust (increase beta) for bootstrapping."""
        if epoch < self.tcfg.bootstrap_start_epoch:
            return
        # Anneal from beta=0.5 → 0.9 over remaining epochs
        progress = (epoch - self.tcfg.bootstrap_start_epoch) / max(
            1, self.tcfg.num_epochs - self.tcfg.bootstrap_start_epoch
        )
        beta = 0.5 + 0.4 * progress
        self.bootstrap_loss.update_beta(beta)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        """Main training loop. Returns the full training history."""
        print("\n" + "=" * 70)
        print("TRAINING START")
        print("=" * 70)

        loader1, loader2 = self.data_module.get_paired_train_loaders()
        val_loader = self.data_module.get_val_loader()
        steps_per_epoch = min(len(loader1), len(loader2))

        # Build schedulers now that we know steps_per_epoch
        self._build_schedulers(steps_per_epoch)

        for epoch in range(self.tcfg.num_epochs):
            epoch_start = time.time()
            print(f"\n[Epoch {epoch + 1}/{self.tcfg.num_epochs}]")

            # Reset epoch-level accumulators
            self.epoch_losses_m1.clear()
            self.epoch_indices.clear()

            # ---- Training ----
            running_loss1, running_loss2, step_count = 0.0, 0.0, 0
            pbar = tqdm(
                zip(loader1, loader2),
                total=steps_per_epoch,
                desc=f"  Train",
                ncols=90,
            )
            for batch1, batch2 in pbar:
                loss1, loss2, info = self._train_step_co_teach(batch1, batch2, epoch)
                running_loss1 += loss1
                running_loss2 += loss2
                step_count += 1

                if step_count % self.tcfg.log_every_n_steps == 0:
                    pbar.set_postfix({
                        "L1": f"{running_loss1/step_count:.4f}",
                        "L2": f"{running_loss2/step_count:.4f}",
                        "FR": f"{info['forget_rate']:.2f}",
                    })

            avg_loss1 = running_loss1 / max(step_count, 1)
            avg_loss2 = running_loss2 / max(step_count, 1)

            # ---- GMM noise analysis ----
            gmm_stats = {}
            if self.tcfg.use_divide_mix and len(self.epoch_losses_m1) > 10:
                gmm_stats = self._run_gmm_analysis(epoch)
                nr = gmm_stats.get("estimated_noise_rate", 0)
                print(f"  [GMM] est. noise={nr:.2%} | "
                      f"clean={gmm_stats.get('num_clean', '?')} | "
                      f"noisy={gmm_stats.get('num_noisy', '?')}")

                # Store full loss array for visualization
                self.all_epoch_loss_history.append(np.array(self.epoch_losses_m1))

            # ---- Bootstrapping annealing ----
            if self.tcfg.use_bootstrapping:
                self._anneal_bootstrapping(epoch)

            # ---- Validation ----
            val_metrics = evaluate_model(
                self.model1, val_loader, self.device,
                cfg=self.cfg, split="val",
            )
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1_weighted"]

            # ---- Logging ----
            keep_ratio = self.small_loss.get_keep_ratio(epoch)
            epoch_time = time.time() - epoch_start
            log_dict = {
                "train_loss": avg_loss1,
                "train_loss_m2": avg_loss2,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "estimated_noise_rate": gmm_stats.get("estimated_noise_rate"),
                "keep_ratio": keep_ratio,
                "forget_rate": self.co_teach.get_forget_rate(epoch),
                "epoch_time": epoch_time,
            }
            self.logger.log_epoch(log_dict)

            print(f"  Loss(M1)={avg_loss1:.4f} | Loss(M2)={avg_loss2:.4f} | "
                  f"Val Acc={val_acc:.4f} | Val F1={val_f1:.4f} | "
                  f"Time={epoch_time:.1f}s")

            # ---- Checkpoint ----
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch + 1
                self._save_checkpoint(epoch)
                print(f"  *** New best model saved (F1={val_f1:.4f}) ***")

        self.logger.save()
        print(f"\n[Training] Done. Best epoch: {self.best_epoch}, "
              f"Best Val F1: {self.best_val_f1:.4f}")
        return self.logger.history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int):
        os.makedirs(self.cfg.model.checkpoint_dir, exist_ok=True)
        ckpt = {
            "epoch": epoch + 1,
            "model1_state": self.model1.state_dict(),
            "model2_state": self.model2.state_dict(),
            "opt1_state": self.opt1.state_dict(),
            "opt2_state": self.opt2.state_dict(),
            "best_val_f1": self.best_val_f1,
            "config": {
                "model_name": self.cfg.model.model_name,
                "num_classes": self.cfg.data.num_classes,
                "noise_type": self.cfg.data.noise_type,
                "noise_rate": self.cfg.data.noise_rate,
            },
        }
        path = os.path.join(self.cfg.model.checkpoint_dir, "best_model.pt")
        torch.save(ckpt, path)

    def load_best_checkpoint(self):
        path = os.path.join(self.cfg.model.checkpoint_dir, "best_model.pt")
        if not os.path.exists(path):
            print("[Checkpoint] No checkpoint found.")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model1.load_state_dict(ckpt["model1_state"])
        self.model2.load_state_dict(ckpt["model2_state"])
        print(f"[Checkpoint] Loaded best model from epoch {ckpt['epoch']} "
              f"(Val F1={ckpt['best_val_f1']:.4f})")
        return ckpt
