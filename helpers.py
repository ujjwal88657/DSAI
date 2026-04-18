"""
utils/helpers.py
General-purpose helper utilities used across the project.
"""

import os
import json
import time
import random
import hashlib
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------
# Device management
# -----------------------------------------------------------------------

def _format_arch_list(arches: List[str]) -> str:
    return ", ".join(arches) if arches else "unknown"


def cuda_compatibility_problem(device_index: int = 0) -> Optional[str]:
    """Return a human-readable CUDA compatibility problem, or None if OK."""
    if not torch.cuda.is_available():
        return "CUDA was requested, but torch.cuda.is_available() is False."

    props = torch.cuda.get_device_properties(device_index)
    major, minor = torch.cuda.get_device_capability(device_index)
    required_arch = f"sm_{major}{minor}"
    compatible_arches = {required_arch, f"compute_{major}{minor}"}

    try:
        compiled_arches = list(torch.cuda.get_arch_list())
    except Exception:
        compiled_arches = []

    if compiled_arches and compatible_arches.isdisjoint(compiled_arches):
        return (
            f"{props.name} has CUDA compute capability {major}.{minor} "
            f"({required_arch}), but this PyTorch build was compiled for: "
            f"{_format_arch_list(compiled_arches)}."
        )

    return None


def cuda_setup_hint(problem: str) -> str:
    return (
        f"{problem}\n\n"
        "This is an environment issue, not a model code issue. Install a PyTorch "
        "build that includes your GPU architecture, choose a newer GPU, or run "
        "with --device cpu."
    )


def get_device(prefer: str = "cuda", strict: bool = False) -> torch.device:
    """Select the best available device and reject incompatible CUDA builds."""
    prefer = (prefer or "auto").lower()
    valid = {"auto", "cuda", "mps", "cpu"}
    if prefer not in valid:
        raise ValueError(f"Unknown device preference '{prefer}'. Expected one of {sorted(valid)}.")

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer in {"cuda", "auto"}:
        if torch.cuda.is_available():
            problem = cuda_compatibility_problem()
            if problem is None:
                return torch.device("cuda")
            if strict or prefer == "cuda":
                raise RuntimeError(cuda_setup_hint(problem))
            print(f"[Device] Skipping CUDA: {problem}")
        elif strict and prefer == "cuda":
            raise RuntimeError(cuda_setup_hint("CUDA was requested, but torch.cuda.is_available() is False."))

    if prefer in {"mps", "auto"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    if strict and prefer == "mps":
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")

    return torch.device("cpu")


def device_info(device: torch.device) -> Dict:
    """Return device metadata dict."""
    info = {"device": str(device)}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        major, minor = torch.cuda.get_device_capability(0)
        info.update({
            "gpu_name": props.name,
            "vram_gb": round(props.total_memory / 1e9, 2),
            "cuda_version": torch.version.cuda,
            "compute_capability": f"{major}.{minor}",
            "torch_cuda_arches": torch.cuda.get_arch_list(),
        })
    return info


# -----------------------------------------------------------------------
# Timing context manager
# -----------------------------------------------------------------------

class Timer:
    """Simple wall-clock timer for profiling."""

    def __init__(self, name: str = ""):
        self.name = name
        self._start = None
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.name:
            print(f"  [{self.name}] {self.elapsed:.3f}s")

    def __str__(self):
        return f"{self.elapsed:.3f}s"


# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------

class JSONLogger:
    """
    Append-mode JSON logger. Each call to log() writes one record.
    Useful for per-step logging without loading the entire file.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        # Write empty list start
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump([], f)

    def log(self, record: Dict):
        with open(self.path, "r") as f:
            data = json.load(f)
        data.append(record)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> List[Dict]:
        with open(self.path) as f:
            return json.load(f)


# -----------------------------------------------------------------------
# Model utilities
# -----------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in megabytes."""
    total_bytes = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)


def freeze_layers(model: torch.nn.Module, layer_names: List[str]):
    """Freeze parameters whose names contain any of the given substrings."""
    frozen = 0
    for name, param in model.named_parameters():
        if any(ln in name for ln in layer_names):
            param.requires_grad = False
            frozen += 1
    print(f"[freeze_layers] Froze {frozen} parameter tensors.")


def unfreeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = True


# -----------------------------------------------------------------------
# Batch utilities
# -----------------------------------------------------------------------

def collate_with_indices(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate that handles variable-length fields."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [item[k] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], str):
            out[k] = vals
        elif isinstance(vals[0], (int, float)):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals
    return out


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move all tensor values in a batch dict to the target device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# -----------------------------------------------------------------------
# EMA (Exponential Moving Average) for model weights
# -----------------------------------------------------------------------

class EMAModel:
    """
    Exponential Moving Average over model weights.
    Maintains a shadow copy of model parameters.
    Used in DivideMix-style training for stable predictions.

    shadow = decay * shadow + (1 - decay) * param
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: torch.nn.Module):
        """Replace model weights with EMA weights (for inference)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        """Restore original weights after shadow inference."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.original:
                param.data.copy_(self.original[name])
        self.original.clear()


# -----------------------------------------------------------------------
# Noise utilities
# -----------------------------------------------------------------------

def compute_true_noise_rate(original_labels: np.ndarray, noisy_labels: np.ndarray) -> float:
    """Compute actual noise rate from original and corrupted labels."""
    assert len(original_labels) == len(noisy_labels)
    return float((original_labels != noisy_labels).mean())


def build_noise_transition_matrix(
    original_labels: np.ndarray,
    noisy_labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Empirically estimate the noise transition matrix T
    where T[i,j] = P(ỹ=j | y=i).
    """
    T = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        mask = original_labels == i
        if mask.sum() == 0:
            T[i, i] = 1.0
            continue
        for j in range(num_classes):
            T[i, j] = (noisy_labels[mask] == j).mean()
    return T


def print_noise_matrix(T: np.ndarray, class_names: List[str]):
    """Pretty-print the noise transition matrix."""
    print("\n  Noise Transition Matrix T[i→j] = P(ỹ=j|y=i):")
    header = "        " + "  ".join(f"{n:>10}" for n in class_names)
    print(header)
    for i, row in enumerate(T):
        vals = "  ".join(f"{v:10.3f}" for v in row)
        print(f"  {class_names[i]:>8}  {vals}")


# -----------------------------------------------------------------------
# Experiment management
# -----------------------------------------------------------------------

def make_run_id(cfg_dict: Optional[Dict] = None) -> str:
    """Generate a unique run ID from config hash + timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg_dict:
        cfg_str = json.dumps(cfg_dict, sort_keys=True)
        h = hashlib.md5(cfg_str.encode()).hexdigest()[:6]
        return f"{ts}_{h}"
    return ts


def save_config(cfg_dict: Dict, output_dir: str, filename: str = "config.json"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"[Config] Saved to {path}")


# -----------------------------------------------------------------------
# Gradient utilities
# -----------------------------------------------------------------------

def get_gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """Compute the total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(norm_type).item() ** norm_type
    return total_norm ** (1.0 / norm_type)


# -----------------------------------------------------------------------
# Label utilities
# -----------------------------------------------------------------------

def labels_to_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer labels to one-hot float tensor."""
    import torch.nn.functional as F
    return F.one_hot(labels, num_classes).float()


def sharpen_probs(probs: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Sharpen probability distribution by reducing temperature."""
    sharpened = probs ** (1.0 / temperature)
    return sharpened / sharpened.sum(dim=1, keepdim=True)


def mixup_labels(
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
    num_classes: int,
) -> torch.Tensor:
    """Create mixed soft labels for MixUp training."""
    import torch.nn.functional as F
    oh_a = F.one_hot(labels_a, num_classes).float()
    oh_b = F.one_hot(labels_b, num_classes).float()
    return lam * oh_a + (1.0 - lam) * oh_b
