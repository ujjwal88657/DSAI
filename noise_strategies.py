"""
training/noise_strategies.py
All noise-robust training strategies from the survey:
  1. Small-Loss Trick
  2. Co-Teaching (Han et al. NeurIPS 2018)
  3. DivideMix-style GMM separation (Li et al. ICLR 2020)
  4. Label Bootstrapping / Pseudo-labeling (Reed et al. ICLR 2015)
  5. MixUp data augmentation
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from typing import Dict, List, Optional, Tuple
from robust_losses import compute_per_sample_loss


# ============================================================
# 1. Small-Loss Trick
# ============================================================

class SmallLossTrick:
    """
    Identifies 'clean' training examples as those with the smallest
    per-sample losses. The intuition is that early in training, DNNs
    first fit the true-labeled examples (which have lower loss).

    The keep ratio is gradually annealed from 1.0 → keep_ratio_final
    to increase noise filtering as training progresses.

    Reference: MentorNet [Jiang et al. ICML 2018], Co-Teaching [Han et al. NeurIPS 2018]
    """

    def __init__(
        self,
        keep_ratio_initial: float = 1.0,
        keep_ratio_final: float = 0.70,
        start_epoch: int = 5,
        total_epochs: int = 20,
    ):
        self.keep_ratio_initial = keep_ratio_initial
        self.keep_ratio_final = keep_ratio_final
        self.start_epoch = start_epoch
        self.total_epochs = total_epochs

    def get_keep_ratio(self, epoch: int) -> float:
        """Linearly anneal keep ratio from initial to final after start_epoch."""
        if epoch < self.start_epoch:
            return self.keep_ratio_initial
        progress = (epoch - self.start_epoch) / max(
            1, self.total_epochs - self.start_epoch
        )
        ratio = self.keep_ratio_initial - progress * (
            self.keep_ratio_initial - self.keep_ratio_final
        )
        return max(ratio, self.keep_ratio_final)

    def select_clean_indices(
        self,
        per_sample_losses: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Return indices of the keep_ratio% lowest-loss examples."""
        keep_ratio = self.get_keep_ratio(epoch)
        n = len(per_sample_losses)
        k = max(1, int(n * keep_ratio))
        _, sorted_indices = per_sample_losses.sort()
        return sorted_indices[:k]

    def filter_batch(
        self,
        batch: Dict[str, torch.Tensor],
        per_sample_losses: torch.Tensor,
        epoch: int,
    ) -> Dict[str, torch.Tensor]:
        """Filter a batch to keep only small-loss examples."""
        clean_idx = self.select_clean_indices(per_sample_losses, epoch)
        return {k: v[clean_idx] if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}


# ============================================================
# 2. Co-Teaching
# ============================================================

class CoTeaching:
    """
    Co-Teaching: two networks train simultaneously. Each network selects
    the small-loss examples from its own perspective and feeds them to
    the PEER network for learning. This reduces confirmation bias.

    forget_rate: fraction of examples to drop (approximately equals noise rate)
    num_gradual: number of epochs to ramp up forget_rate (from 0 → forget_rate)

    Reference: "Co-teaching: Robust Training of Deep Neural Networks with
    Extremely Noisy Labels" Han et al., NeurIPS 2018.
    """

    def __init__(
        self,
        forget_rate: float = 0.20,
        num_gradual: int = 10,
        total_epochs: int = 20,
        exponent: float = 1.0,
    ):
        self.forget_rate = forget_rate
        self.num_gradual = num_gradual
        self.total_epochs = total_epochs
        self.exponent = exponent
        # Pre-compute rate schedule
        self._rate_schedule = self._build_schedule()

    def _build_schedule(self) -> np.ndarray:
        """
        Gradually increase the forget rate from 0 to forget_rate
        over num_gradual epochs. Stays at forget_rate after that.
        """
        schedule = np.ones(self.total_epochs) * self.forget_rate
        schedule[:self.num_gradual] = np.linspace(
            0, self.forget_rate ** self.exponent, self.num_gradual
        )
        return schedule

    def get_forget_rate(self, epoch: int) -> float:
        idx = min(epoch, len(self._rate_schedule) - 1)
        return float(self._rate_schedule[idx])

    def select_indices(
        self,
        losses: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Select indices of examples to KEEP (lowest-loss)."""
        forget_rate = self.get_forget_rate(epoch)
        n = len(losses)
        num_keep = max(1, int(n * (1 - forget_rate)))
        _, sorted_idx = losses.sort()
        return sorted_idx[:num_keep]

    def co_teach_step(
        self,
        model1,
        model2,
        batch1: Dict[str, torch.Tensor],
        batch2: Dict[str, torch.Tensor],
        loss_fn,
        epoch: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        One Co-Teaching training step.
        - model1 sees batch2's selected clean examples
        - model2 sees batch1's selected clean examples
        Returns losses for model1 and model2, plus diagnostic info.
        """
        def _get_batch_tensors(batch):
            return (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
                batch["label"].to(device),
            )

        ids1, mask1, tt1, lbl1 = _get_batch_tensors(batch1)
        ids2, mask2, tt2, lbl2 = _get_batch_tensors(batch2)

        # ------ Model 1 selects clean from batch1 ------
        with torch.no_grad():
            out1 = model1(ids1, mask1, tt1)
        losses1 = compute_per_sample_loss(out1["logits"], lbl1)
        clean_idx1 = self.select_indices(losses1, epoch)

        # ------ Model 2 selects clean from batch2 ------
        with torch.no_grad():
            out2 = model2(ids2, mask2, tt2)
        losses2 = compute_per_sample_loss(out2["logits"], lbl2)
        clean_idx2 = self.select_indices(losses2, epoch)

        # ------ Model 1 trains on batch1's clean examples selected BY model2 ------
        # (model2's selection from batch2 used on batch1 — same size via indexing)
        # Standard Co-Teaching: network i trains on examples selected by network j
        # from the SAME batch (using separate shuffled loaders ensures independence)
        # We keep it simple: model i selects from its batch, trains the peer on those.
        out1_train = model1(ids2[clean_idx2], mask2[clean_idx2], tt2[clean_idx2])
        loss1 = loss_fn(out1_train["logits"], lbl2[clean_idx2]).mean()

        out2_train = model2(ids1[clean_idx1], mask1[clean_idx1], tt1[clean_idx1])
        loss2 = loss_fn(out2_train["logits"], lbl1[clean_idx1]).mean()

        info = {
            "num_clean_1": len(clean_idx1),
            "num_clean_2": len(clean_idx2),
            "forget_rate": self.get_forget_rate(epoch),
        }
        return loss1, loss2, info


# ============================================================
# 3. DivideMix-style GMM
# ============================================================

class GaussianMixtureNoiseSeparator:
    """
    Fits a 2-component 1D Gaussian Mixture Model to the per-sample loss
    distribution to probabilistically separate clean and noisy examples.

    The component with the SMALLER mean is treated as the 'clean' component.
    Each example receives a probability p_clean of being correctly labeled.

    References:
      - DivideMix: Li et al. ICLR 2020
      - Dynamic Bootstrapping: Arazo et al. ICML 2019
    """

    def __init__(self, p_threshold: float = 0.5):
        self.p_threshold = p_threshold
        self.mu = None        # component means
        self.sigma = None     # component stds
        self.pi = None        # mixing weights

    def _em_step(
        self,
        losses: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        pi: np.ndarray,
        n_iter: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple EM algorithm for 2-component 1D GMM."""
        losses = losses.reshape(-1)
        for _ in range(n_iter):
            # E-step
            p0 = pi[0] * norm.pdf(losses, mu[0], sigma[0]) + 1e-10
            p1 = pi[1] * norm.pdf(losses, mu[1], sigma[1]) + 1e-10
            r0 = p0 / (p0 + p1)
            r1 = 1.0 - r0

            # M-step
            n0, n1 = r0.sum(), r1.sum()
            mu[0] = (r0 * losses).sum() / (n0 + 1e-10)
            mu[1] = (r1 * losses).sum() / (n1 + 1e-10)
            sigma[0] = np.sqrt((r0 * (losses - mu[0]) ** 2).sum() / (n0 + 1e-10)) + 1e-6
            sigma[1] = np.sqrt((r1 * (losses - mu[1]) ** 2).sum() / (n1 + 1e-10)) + 1e-6
            pi[0] = n0 / len(losses)
            pi[1] = n1 / len(losses)

        return mu, sigma, pi

    def fit_predict(
        self,
        losses: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fit GMM to loss array and return:
          - p_clean: per-sample probability of being clean
          - is_clean: boolean mask (p_clean > p_threshold)
          - estimated noise rate
        """
        losses_np = losses.reshape(-1)

        # Initialize: split at median
        median = np.median(losses_np)
        mu = np.array([losses_np[losses_np <= median].mean(),
                       losses_np[losses_np > median].mean()])
        sigma = np.array([losses_np[losses_np <= median].std() + 1e-6,
                          losses_np[losses_np > median].std() + 1e-6])
        pi = np.array([0.5, 0.5])

        mu, sigma, pi = self._em_step(losses_np, mu, sigma, pi)

        # Ensure component 0 = clean (smaller mean)
        if mu[0] > mu[1]:
            mu = mu[[1, 0]]
            sigma = sigma[[1, 0]]
            pi = pi[[1, 0]]

        self.mu, self.sigma, self.pi = mu, sigma, pi

        # Compute posterior probabilities p(clean | loss)
        p_clean_num = pi[0] * norm.pdf(losses_np, mu[0], sigma[0]) + 1e-10
        p_total = (
            pi[0] * norm.pdf(losses_np, mu[0], sigma[0])
            + pi[1] * norm.pdf(losses_np, mu[1], sigma[1])
            + 1e-10
        )
        p_clean = p_clean_num / p_total
        is_clean = p_clean > self.p_threshold
        estimated_noise_rate = 1.0 - is_clean.mean()

        return p_clean, is_clean, float(estimated_noise_rate)

    def get_stats(self) -> Dict:
        if self.mu is None:
            return {}
        return {
            "mu_clean": float(self.mu[0]),
            "mu_noisy": float(self.mu[1]),
            "sigma_clean": float(self.sigma[0]),
            "sigma_noisy": float(self.sigma[1]),
            "pi_clean": float(self.pi[0]),
        }


# ============================================================
# 4. Label Refurbishment Store
# ============================================================

class LabelRefurbishmentStore:
    """
    Maintains soft label predictions accumulated over training.
    Implements:
      - Pseudo-labeling: replace noisy label with model's argmax prediction
      - Soft bootstrapping: blend noisy label with model's soft output
      - Exponential moving average (EMA) of predictions for stability

    Reference:
      - SELFIE [Song et al. ICML 2019]
      - Self-Adaptive Training [Huang et al. NeurIPS 2020]
    """

    def __init__(
        self,
        n_samples: int,
        num_classes: int,
        alpha: float = 0.9,    # EMA smoothing factor
        device: str = "cpu",
    ):
        self.n_samples = n_samples
        self.num_classes = num_classes
        self.alpha = alpha
        self.device = device

        # Soft prediction accumulators
        self.ema_preds = torch.ones(n_samples, num_classes).float() / num_classes
        self.prediction_counts = torch.zeros(n_samples).long()

    def update(self, indices: torch.Tensor, soft_preds: torch.Tensor):
        """
        Update EMA of soft predictions for given sample indices.
        soft_preds: (B, C) softmax probabilities
        """
        indices_cpu = indices.cpu()
        preds_cpu = soft_preds.detach().cpu().float()
        self.ema_preds[indices_cpu] = (
            self.alpha * self.ema_preds[indices_cpu]
            + (1.0 - self.alpha) * preds_cpu
        )
        self.prediction_counts[indices_cpu] += 1

    def get_corrected_labels(
        self,
        indices: torch.Tensor,
        noisy_labels: torch.Tensor,
        beta: float = 0.8,
        mode: str = "soft",       # "soft" | "hard" | "ema"
    ) -> torch.Tensor:
        """
        Return corrected labels for a batch.
        mode:
          - "hard": pure pseudo-label (argmax of EMA prediction)
          - "soft": beta * one_hot(noisy) + (1-beta) * EMA_pred
          - "ema":  pure EMA soft labels
        """
        idx_cpu = indices.cpu()
        ema = self.ema_preds[idx_cpu].to(noisy_labels.device)

        if mode == "hard":
            return ema.argmax(dim=1)
        elif mode == "ema":
            return ema
        else:  # "soft"
            one_hot = F.one_hot(noisy_labels, self.num_classes).float()
            return beta * one_hot + (1.0 - beta) * ema

    def has_enough_predictions(self, min_updates: int = 3) -> bool:
        """Check if enough EMA updates have been collected."""
        return bool((self.prediction_counts >= min_updates).all())


# ============================================================
# 5. MixUp Augmentation
# ============================================================

def mixup_data(
    x: Dict[str, torch.Tensor],
    y: torch.Tensor,
    alpha: float = 4.0,
    num_classes: int = 3,
    device: torch.device = None,
) -> Tuple[Dict, torch.Tensor, torch.Tensor, float]:
    """
    MixUp data augmentation applied to token embeddings and soft labels.
    Since we work at the hidden-state level (not raw pixels), MixUp is
    applied AFTER the first forward pass of the encoder.

    y_mix = lambda * y_a + (1-lambda) * y_b

    Reference: "mixup: Beyond Empirical Risk Minimization" Zhang et al., ICLR 2018.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    lam = max(lam, 1.0 - lam)   # Ensure lam >= 0.5 for stability

    batch_size = y.size(0)
    perm_idx = torch.randperm(batch_size)

    # Soft one-hot targets
    y_oh = F.one_hot(y, num_classes).float()
    y_b = y_oh[perm_idx]
    y_mix = lam * y_oh + (1.0 - lam) * y_b

    return perm_idx, y_mix, lam


def mixup_embeddings(
    embeddings: torch.Tensor,
    perm_idx: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Mix embedding vectors for MixUp."""
    return lam * embeddings + (1.0 - lam) * embeddings[perm_idx]


def soft_ce_loss(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy with soft (non-one-hot) targets."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_labels * log_probs).sum(dim=1).mean()


# ============================================================
# 6. Noise Rate Estimator
# ============================================================

class NoiseRateEstimator:
    """
    Estimates the current effective noise rate using the GMM-based
    separator. Provides running statistics for logging.
    """

    def __init__(self):
        self.history: List[float] = []
        self.gmm = GaussianMixtureNoiseSeparator()

    def estimate(self, losses: np.ndarray, threshold: float = 0.5) -> Dict:
        """Estimate noise rate from current epoch's loss distribution."""
        p_clean, is_clean, noise_rate = self.gmm.fit_predict(losses)
        self.history.append(noise_rate)
        stats = self.gmm.get_stats()
        stats.update({
            "estimated_noise_rate": noise_rate,
            "num_clean": int(is_clean.sum()),
            "num_noisy": int((~is_clean).sum()),
        })
        return stats, p_clean, is_clean
