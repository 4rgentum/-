"""Loss functions: BCE with logits, Focal loss (Lin et al., 2017)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    r"""Binary focal loss.

    .. math::
        L = -\alpha_t (1 - p_t)^\gamma \log p_t

    Args:
        alpha:  weight of the positive class (gives extra mass to attacks).
        gamma:  focusing parameter; larger -> more focus on hard examples.
        reduction: ``mean`` | ``sum`` | ``none``.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        if weights is not None:
            loss = loss * weights
        return self._reduce(loss)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return x.mean()
        if self.reduction == "sum":
            return x.sum()
        return x


class BCEWithLogits(nn.Module):
    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        self.pos_weight = (
            torch.tensor([float(pos_weight)]) if pos_weight is not None else None
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pw)


def build_loss(loss_cfg: dict[str, Any], y_train: torch.Tensor | None = None) -> nn.Module:
    name = loss_cfg.get("name", "bce")
    if name == "focal":
        return FocalLoss(
            alpha=float(loss_cfg.get("alpha", 0.25)),
            gamma=float(loss_cfg.get("gamma", 2.0)),
        )
    if name == "bce":
        pos_weight = loss_cfg.get("class_weights")
        if pos_weight == "auto" and y_train is not None:
            n_pos = max(int((y_train == 1).sum()), 1)
            n_neg = max(int((y_train == 0).sum()), 1)
            pos_weight = float(n_neg / n_pos)
        elif isinstance(pos_weight, (list, tuple)) and len(pos_weight) == 2:
            pos_weight = float(pos_weight[1] / pos_weight[0])
        else:
            pos_weight = None
        return BCEWithLogits(pos_weight=pos_weight)
    raise ValueError(f"unknown loss: {name}")
