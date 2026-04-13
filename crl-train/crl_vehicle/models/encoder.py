"""
MultiTaskEncoder

Maps SSM output (B, T', d_model) to three task-specific embeddings via
separate projection branches. This eliminates gradient competition between
differently-sized latent blocks (e.g. 1-dim presence vs 8-dim instance).

    e_pres  (d_pres)  — vehicle presence embedding
    e_type  (d_type)  — vehicle type embedding
    e_inst  (d_inst)  — vehicle instance embedding

Each branch receives the same attention-pooled SSM context vector but has
independent weights. Single linear projections (no ReLU) keep embeddings
linearly separable for downstream probe training.

CRLHeads: lightweight classification heads used only during CRL training to
provide task supervision. These are discarded after backbone training.
"""

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """
    Soft attention pooling over a sequence dimension.
    Linear(d_model → 1) → softmax over T' → weighted sum → (B, d_model).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T', d_model)
        Returns: (B, d_model) context vector
        """
        attn = torch.softmax(self.score(x), dim=1)   # (B, T', 1)
        return (attn * x).sum(dim=1)                 # (B, d_model)


class MultiTaskEncoder(nn.Module):
    """
    Branched encoder producing three independent task embeddings.

    Args:
        d_model : SSM output dimension (= CRLConfig.d_model)
        d_pres  : presence embedding dimension
        d_type  : type embedding dimension
        d_inst  : instance embedding dimension
    """

    def __init__(
        self,
        d_model: int,
        d_pres: int = 16,
        d_type: int = 32,
        d_inst: int = 64,
    ):
        super().__init__()
        self.d_pres = d_pres
        self.d_type = d_type
        self.d_inst = d_inst

        self.attn_pool = AttentionPool(d_model)
        self.proj_pres = nn.Linear(d_model, d_pres)
        self.proj_type = nn.Linear(d_model, d_type)
        self.proj_inst = nn.Linear(d_model, d_inst)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T', d_model) — SSM output

        Returns:
            e_pres : (B, d_pres) presence embedding
            e_type : (B, d_type) type embedding
            e_inst : (B, d_inst) instance embedding
        """
        h = self.attn_pool(x)                # (B, d_model)
        return self.proj_pres(h), self.proj_type(h), self.proj_inst(h)


class CRLHeads(nn.Module):
    """
    Lightweight classification heads used only during CRL backbone training.
    Discarded after CRL phase — downstream uses VehicleDetectionHead instead.

    These are kept separate from downstream heads so that:
    1. Downstream heads can be reset and retrained without affecting backbone.
    2. CRL training uses simpler linear heads (lower risk of overfitting).

    Args:
        d_pres      : presence embedding dimension
        d_type      : type embedding dimension
        d_inst      : instance embedding dimension
        n_type      : number of vehicle type classes (default 4)
        n_inst      : number of vehicle instance classes (default 13)
    """

    def __init__(
        self,
        d_pres: int = 16,
        d_type: int = 32,
        d_inst: int = 64,
        n_type: int = 4,
        n_inst: int = 13,
    ):
        super().__init__()
        self.pres_cls = nn.Linear(d_pres, 1)        # BCE: vehicle present?
        self.type_cls = nn.Linear(d_type, n_type)   # CE: which type?
        self.inst_cls = nn.Linear(d_inst, n_inst)   # CE: which instance?

    def forward(
        self,
        e_pres: torch.Tensor,
        e_type: torch.Tensor,
        e_inst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        e_pres: (B, d_pres)
        e_type: (B, d_type)
        e_inst: (B, d_inst)

        Returns:
            pres_logit  : (B,)        for BCEWithLogitsLoss
            type_logits : (B, n_type) for CrossEntropyLoss
            inst_logits : (B, n_inst) for CrossEntropyLoss
        """
        return (
            self.pres_cls(e_pres).squeeze(-1),
            self.type_cls(e_type),
            self.inst_cls(e_inst),
        )
