"""
models/classifier.py
BERT-based text classifier with flexible classification head.
Supports both single-model and dual-model (Co-Teaching) configurations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple


class AttentionPooling(nn.Module):
    """
    Attention-weighted mean pooling over token hidden states.
    More expressive than simple [CLS] or mean pooling.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,     # (B, L, H)
        attention_mask: torch.Tensor,    # (B, L)
    ) -> torch.Tensor:                   # (B, H)
        scores = self.attention(hidden_states).squeeze(-1)  # (B, L)
        # Mask padding tokens
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)   # (B, L, 1)
        pooled = (hidden_states * weights).sum(dim=1)        # (B, H)
        return pooled


class ClassificationHead(nn.Module):
    """
    Multi-layer classification head with dropout and optional residual.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        num_classes: int,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class BERTClassifier(nn.Module):
    """
    Complete BERT-based classifier for Hinglish content moderation.

    Architecture:
        BERT encoder → Attention pooling → Classification head → logits

    The encoder can be frozen during early epochs for efficiency.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg.model

        # Load pretrained BERT
        print(f"[Model] Loading backbone: {mcfg.model_name}")
        bert_config = AutoConfig.from_pretrained(
            mcfg.model_name,
            output_hidden_states=True,
        )
        self.bert = AutoModel.from_pretrained(mcfg.model_name, config=bert_config)

        # Pooling
        self.pooler = AttentionPooling(mcfg.hidden_size)

        # Dropout before head
        self.dropout = nn.Dropout(mcfg.dropout_rate)

        # Classification head
        self.head = ClassificationHead(
            input_dim=mcfg.hidden_size,
            hidden_dims=mcfg.classifier_hidden_dims,
            num_classes=mcfg.num_classes,
            dropout_rate=mcfg.dropout_rate,
        )

        self._init_head_weights()

    def _init_head_weights(self):
        """Initialize classification head weights with small values."""
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def freeze_encoder(self, num_layers: Optional[int] = None):
        """
        Freeze BERT parameters.
        If num_layers is specified, only freeze the first num_layers transformer layers.
        """
        if num_layers is None:
            for p in self.bert.parameters():
                p.requires_grad = False
            print("[Model] Encoder fully frozen.")
        else:
            # Freeze embeddings
            for p in self.bert.embeddings.parameters():
                p.requires_grad = False
            # Freeze first num_layers encoder layers
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < num_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
            print(f"[Model] Froze first {num_layers} encoder layers + embeddings.")

    def unfreeze_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = True
        print("[Model] Encoder unfrozen.")

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return pooled representations (before classification head). Used for visualization."""
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            hidden_states = outputs.last_hidden_state
            pooled = self.pooler(hidden_states, attention_mask)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = outputs.last_hidden_state     # (B, L, H)
        pooled = self.pooler(hidden_states, attention_mask)   # (B, H)
        dropped = self.dropout(pooled)
        logits = self.head(dropped)                   # (B, C)

        result = {"logits": logits}
        if return_embeddings:
            result["embeddings"] = pooled
        return result

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_model(cfg, device: torch.device) -> BERTClassifier:
    """Instantiate and move model to device."""
    model = BERTClassifier(cfg)
    model = model.to(device)
    params = model.count_parameters()
    print(f"[Model] Total params: {params['total']:,} | Trainable: {params['trainable']:,}")
    return model


def build_dual_models(cfg, device: torch.device) -> Tuple[BERTClassifier, BERTClassifier]:
    """
    Build two independent BERT classifiers for Co-Teaching.
    Both share the same architecture but are initialized differently
    (due to random dropout and head init).
    """
    model1 = build_model(cfg, device)
    model2 = build_model(cfg, device)
    print("[Model] Dual models built for Co-Teaching.")
    return model1, model2
