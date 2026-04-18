"""
losses/robust_losses.py
All noise-robust loss functions:
  - CrossEntropy (baseline)
  - SymmetricCrossEntropy (SCE)
  - GeneralizedCrossEntropy (GCE)
  - MeanAbsoluteError (MAE) loss
  - BootstrappingLoss (label correction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

EPS = 1e-7


def _onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert label indices to one-hot float tensors."""
    return F.one_hot(labels, num_classes).float()


# ---------------------------------------------------------------------------
# Standard Cross-Entropy (baseline)
# ---------------------------------------------------------------------------

class CrossEntropyLoss(nn.Module):
    """Standard CE loss — susceptible to noisy labels."""

    def __init__(self, num_classes: int, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.ce(logits, labels)
        if weights is not None:
            loss = loss * weights
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # none


# ---------------------------------------------------------------------------
# Symmetric Cross-Entropy (SCE)  [Wang et al. ICCV 2019]
# ---------------------------------------------------------------------------

class SymmetricCrossEntropyLoss(nn.Module):
    """
    SCE = alpha * CE(p, q) + beta * RCE(q, p)

    Where:
      - CE  is the standard cross-entropy (forward direction)
      - RCE is the reverse cross-entropy (backward direction)

    RCE is noise-tolerant because it does NOT require the model to be
    overconfident about noisy predictions.

    Paper: "Symmetric Cross Entropy for Robust Learning with Noisy Labels"
    Wang et al., ICCV 2019.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.1,
        beta: float = 1.0,
        reduction: str = "mean",
        A: float = -6.0,     # numerical clamp for RCE (log(eps))
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.A = A
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # --- Forward CE ---
        ce_loss = self.ce(logits, labels)

        # --- Reverse CE ---
        pred = F.softmax(logits, dim=1)
        pred = torch.clamp(pred, min=EPS, max=1.0)

        label_oh = _onehot(labels, self.num_classes).to(logits.device)
        # RCE = -sum_j [ p(y=j|x) * log(q(y=j)) ]
        # where q is the one-hot target
        # Clamp log of target to avoid log(0)
        log_target = torch.clamp(label_oh, min=EPS).log()
        log_target = torch.clamp(log_target, min=self.A)   # numerical stability
        rce_loss = -torch.sum(pred * log_target, dim=1)

        loss = self.alpha * ce_loss + self.beta * rce_loss

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # "none"


# ---------------------------------------------------------------------------
# Generalized Cross-Entropy (GCE)  [Zhang & Sabuncu NeurIPS 2018]
# ---------------------------------------------------------------------------

class GeneralizedCrossEntropyLoss(nn.Module):
    """
    GCE(f, y) = (1 - f_y^q) / q

    Interpolates between MAE (q→0) and CE (q→1).
    q=0.7 is recommended by the original paper.

    Paper: "Generalized Cross Entropy Loss for Training Deep Neural Networks
    with Noisy Labels", Zhang & Sabuncu, NeurIPS 2018.
    """

    def __init__(self, num_classes: int, q: float = 0.7, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.q = q
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred = F.softmax(logits, dim=1)
        pred = torch.clamp(pred, min=EPS)

        # Gather predicted probability for the correct class
        label_oh = _onehot(labels, self.num_classes).to(logits.device)
        p_y = (pred * label_oh).sum(dim=1)               # shape: (B,)

        loss = (1.0 - p_y ** self.q) / self.q

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Mean Absolute Error (MAE) Loss  [Ghosh et al. AAAI 2017]
# ---------------------------------------------------------------------------

class MAELoss(nn.Module):
    """
    MAE loss for classification:
    L_MAE(f, y) = 2 - 2 * f_y

    Theoretically noise-tolerant under symmetric noise for any τ < (c-1)/c.
    Slower convergence than CE but robust.

    Paper: "Robust Loss Functions under Label Noise for Deep Neural Networks"
    Ghosh et al., AAAI 2017.
    """

    def __init__(self, num_classes: int, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred = F.softmax(logits, dim=1)
        label_oh = _onehot(labels, self.num_classes).to(logits.device)
        loss = 1.0 - (pred * label_oh).sum(dim=1)         # = 1 - p_y

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Bootstrapping / Label Correction Loss  [Reed et al. ICLR 2015]
# ---------------------------------------------------------------------------

class BootstrappingLoss(nn.Module):
    """
    Soft bootstrapping:
      y_corrected = beta * y_noisy + (1 - beta) * f(x)

    The refurbished label blends the noisy annotation with the model's
    current prediction to gradually reduce the influence of incorrect labels.

    Paper: "Training Deep Neural Networks on Noisy Labels with Bootstrapping"
    Reed et al., ICLR 2015.
    """

    def __init__(
        self,
        num_classes: int,
        beta: float = 0.8,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred = F.softmax(logits, dim=1)
        label_oh = _onehot(labels, self.num_classes).to(logits.device)

        # Soft refurbished target
        corrected = self.beta * label_oh + (1.0 - self.beta) * pred.detach()

        # Cross-entropy with soft target
        log_pred = torch.log(pred + EPS)
        loss = -torch.sum(corrected * log_pred, dim=1)

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def update_beta(self, new_beta: float):
        """Anneal beta over training (decrease model trust in noisy labels)."""
        self.beta = new_beta


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def build_loss(cfg) -> nn.Module:
    """Construct the primary loss function from config."""
    loss_type = cfg.training.loss_type
    num_classes = cfg.data.num_classes

    if loss_type == "ce":
        return CrossEntropyLoss(num_classes=num_classes, reduction="none")
    elif loss_type == "sce":
        return SymmetricCrossEntropyLoss(
            num_classes=num_classes,
            alpha=cfg.training.sce_alpha,
            beta=cfg.training.sce_beta,
            reduction="none",
        )
    elif loss_type == "gce":
        return GeneralizedCrossEntropyLoss(
            num_classes=num_classes,
            q=cfg.training.gce_q,
            reduction="none",
        )
    elif loss_type == "mae":
        return MAELoss(num_classes=num_classes, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# ---------------------------------------------------------------------------
# Per-sample loss utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_per_sample_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sample CE loss without reduction.
    Used for small-loss trick, DivideMix GMM fitting, and Co-Teaching selection.
    """
    loss = F.cross_entropy(logits, labels, reduction="none")
    return loss
