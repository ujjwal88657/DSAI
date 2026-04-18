"""
config.py - Central configuration for the entire project.
All hyperparameters, paths, and settings live here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    # Dataset settings
    dataset_name: str = "combined_hate_speech"   # identifier for the dataset
    dataset_path: str = "./combined_hate_speech_dataset.csv"
    text_column: str = "text"
    label_column: str = "hate_label"
    data_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    num_classes: int = 2                          # inferred again after CSV load
    class_names: List[str] = field(default_factory=lambda: ["non_hate", "hate"])
    use_cache: bool = False                       # always read the source CSV by default

    # Noise simulation
    simulate_noise: bool = True
    noise_type: str = "asymmetric"               # symmetric | asymmetric | instance
    noise_rate: float = 0.30                     # 30% label corruption
    noise_seed: int = 42

    # Splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # Tokenizer
    max_seq_len: int = 128
    tokenizer_name: str = "bert-base-multilingual-cased"


@dataclass
class ModelConfig:
    # Backbone
    model_name: str = "bert-base-multilingual-cased"
    hidden_size: int = 768
    num_classes: int = 3
    dropout_rate: float = 0.3

    # Classification head architecture
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [256])

    # Checkpoints
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True


@dataclass
class TrainingConfig:
    # Core training
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    device: str = "cuda"                         # auto-detected in main

    # Dual model (Co-Teaching)
    use_co_teaching: bool = True
    forget_rate: float = 0.20                    # fraction of large-loss examples to drop
    num_gradual: int = 10                        # epochs to gradually increase forget rate
    exponent: float = 1.0

    # DivideMix
    use_divide_mix: bool = True
    alpha: float = 4.0                           # Beta distribution parameter for MixUp
    p_threshold: float = 0.5                     # GMM clean probability threshold
    temperature: float = 0.5                     # sharpening temperature

    # Label correction / Bootstrapping
    use_bootstrapping: bool = True
    bootstrap_beta: float = 0.8                  # weight on original label vs. model pred
    bootstrap_start_epoch: int = 5               # epoch to start bootstrapping

    # Small-loss trick
    use_small_loss: bool = True
    small_loss_start_epoch: int = 5
    keep_ratio_initial: float = 1.0
    keep_ratio_final: float = 0.70               # keep top 70% small-loss examples

    # Loss function
    loss_type: str = "sce"                       # ce | sce | gce | mae
    sce_alpha: float = 0.1                       # SCE alpha (CE weight)
    sce_beta: float = 1.0                        # SCE beta (reverse CE weight)
    gce_q: float = 0.7                           # GCE q parameter

    # MixMatch semi-supervised
    use_mixmatch: bool = False                   # disabled by default for speed
    mixmatch_lambda_u: float = 75.0
    mixmatch_T: float = 0.5
    mixmatch_K: int = 2                          # augmentations per unlabeled example

    # Logging
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 1
    log_dir: str = "./logs"


@dataclass
class VisualizationConfig:
    output_dir: str = "./visualizations"
    dpi: int = 150
    fig_format: str = "png"
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    pca_n_components: int = 2
    embedding_sample_size: int = 500             # samples for embedding viz
    color_palette: str = "tab10"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        # Sync num_classes across sub-configs
        self.model.num_classes = self.data.num_classes
        # Create directories
        os.makedirs(self.data.data_dir, exist_ok=True)
        os.makedirs(self.data.processed_dir, exist_ok=True)
        os.makedirs(self.model.checkpoint_dir, exist_ok=True)
        os.makedirs(self.training.log_dir, exist_ok=True)
        os.makedirs(self.viz.output_dir, exist_ok=True)

    def display(self):
        import json
        from dataclasses import asdict
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(json.dumps(asdict(self), indent=2))
        print("=" * 60)


# Global config instance
CFG = Config()
