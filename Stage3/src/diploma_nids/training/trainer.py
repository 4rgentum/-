"""DL trainer with early stopping, schedulers and metric monitoring."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..eval.metrics import classification_metrics, integral_metrics
from ..models.base import DLDetector
from ..utils.logging import get_logger
from .losses import build_loss

logger = get_logger(__name__)


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_pr_auc: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)


def _device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


def _build_optimizer(params, opt_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = opt_cfg.get("name", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
        )
    if name == "adam":
        return torch.optim.Adam(params, lr=float(opt_cfg["lr"]), weight_decay=float(opt_cfg.get("weight_decay", 0.0)))
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(opt_cfg["lr"]),
            momentum=float(opt_cfg.get("momentum", 0.9)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        )
    raise ValueError(f"unknown optimizer: {name}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer, sch_cfg: dict[str, Any], total_epochs: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = sch_cfg.get("name", "cosine").lower() if sch_cfg else None
    if not name:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=float(sch_cfg.get("min_lr", 1e-6))
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(sch_cfg["step_size"]), gamma=float(sch_cfg.get("gamma", 0.1))
        )
    return None


class Trainer:
    """Train a DL detector and return the best-checkpoint state-dict."""

    def __init__(self, train_cfg: dict[str, Any]) -> None:
        self.cfg = train_cfg
        self.tcfg = train_cfg["trainer"]
        self.opt_cfg = train_cfg["optimizer"]
        self.sch_cfg = train_cfg.get("scheduler", {})
        self.loss_cfg = train_cfg["loss"]
        self.device = _device(self.tcfg.get("device", "auto"))

    def fit(
        self,
        model: DLDetector,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        ckpt_path: str | Path | None = None,
    ) -> TrainHistory:
        model.to(self.device)
        loss_fn = build_loss(self.loss_cfg, torch.from_numpy(y_train.astype(np.float32)))
        loss_fn.to(self.device)
        optimizer = _build_optimizer(model.parameters(), self.opt_cfg)
        epochs = int(self.tcfg["epochs"])
        scheduler = _build_scheduler(optimizer, self.sch_cfg, epochs)

        train_loader = _make_loader(
            X_train, y_train,
            batch_size=int(self.tcfg["batch_size"]),
            shuffle=True,
            num_workers=int(self.tcfg.get("num_workers", 0)),
        )
        val_loader = _make_loader(
            X_val, y_val,
            batch_size=int(self.tcfg["batch_size"]),
            shuffle=False,
            num_workers=int(self.tcfg.get("num_workers", 0)),
        )

        history = TrainHistory()
        monitor = str(self.tcfg.get("early_stopping_metric", "val_pr_auc"))
        mode = str(self.tcfg.get("monitor_mode", "max"))
        patience = int(self.tcfg.get("early_stopping_patience", 10))
        best = -math.inf if mode == "max" else math.inf
        bad = 0
        best_state: dict[str, torch.Tensor] | None = None

        grad_clip = float(self.tcfg.get("grad_clip", 0.0))

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out.logits, yb)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                train_losses.append(float(loss.item()))

            # Validation
            model.eval()
            val_losses, val_logits, val_targets = [], [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    out = model(xb)
                    val_losses.append(float(loss_fn(out.logits, yb).item()))
                    val_logits.append(out.logits.detach().cpu().numpy())
                    val_targets.append(yb.detach().cpu().numpy())

            val_score = 1.0 / (1.0 + np.exp(-np.concatenate(val_logits)))
            val_y = np.concatenate(val_targets).astype(int)
            val_pred = (val_score >= 0.5).astype(int)
            cls = classification_metrics(val_y, val_pred)
            integ = integral_metrics(val_y, val_score)

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.val_pr_auc.append(integ["pr_auc"])
            history.val_f1.append(cls.f1)

            metric_value = {"val_pr_auc": integ["pr_auc"], "val_f1": cls.f1, "val_loss": val_loss}[monitor]
            improved = (mode == "max" and metric_value > best) or (mode == "min" and metric_value < best)
            if improved:
                best = metric_value
                bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if ckpt_path:
                    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state_dict": best_state, "epoch": epoch, monitor: metric_value}, ckpt_path)
            else:
                bad += 1

            if scheduler is not None:
                scheduler.step()

            logger.info(
                "epoch %3d | train_loss=%.4f val_loss=%.4f val_pr_auc=%.4f val_f1=%.4f best=%s%.4f",
                epoch, train_loss, val_loss, integ["pr_auc"], cls.f1, "+" if improved else "", best,
            )
            if bad >= patience:
                logger.info("early stop at epoch %d (no improvement for %d epochs)", epoch, patience)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return history
