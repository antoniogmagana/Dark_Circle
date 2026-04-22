from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(
    anchor: torch.Tensor,
    partners: torch.Tensor,
    is_positive: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """NT-Xent (SimCLR) loss over stratified partners.

    anchor:      (B, D) — L2-normalized anchor embeddings.
    partners:    (B, P, D) — L2-normalized partner embeddings.
    is_positive: (B, P) bool — True where partner p for anchor b is a positive.
                 Negatives are the remaining partners for the same anchor PLUS
                 all partners belonging to *other* anchors in the batch (other
                 anchors' positives are treated as in-batch negatives, standard
                 SimCLR practice).
    temperature: softmax temperature.

    Returns a scalar loss. Rows with zero positives are dropped from the mean.
    """
    B, P, D = partners.shape
    flat = partners.reshape(B * P, D)  # (B*P, D)
    # logits[b, k] = <anchor_b, partner_k> / temperature, k indexes B*P
    logits = (anchor @ flat.T) / temperature  # (B, B*P)

    # Positive mask in flat index space: (B, B*P)
    pos_mask = torch.zeros(B, B * P, dtype=torch.bool, device=anchor.device)
    own_cols = torch.arange(P, device=anchor.device).unsqueeze(0) + \
               torch.arange(B, device=anchor.device).unsqueeze(1) * P
    pos_mask.scatter_(1, own_cols, is_positive)

    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return torch.zeros((), device=anchor.device, requires_grad=True)

    logits = logits[has_pos]
    pos_mask = pos_mask[has_pos]

    # log p(positive | all) — sum log-probs over positives, average per anchor.
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    n_pos = pos_mask.sum(dim=1).clamp(min=1)
    per_anchor = -(log_prob * pos_mask).sum(dim=1) / n_pos
    return per_anchor.mean()
