"""FastAPI inference service.

Endpoints:
    GET  /health
    GET  /info
    POST /score          (body = {"records": [{...}, ...]})
    POST /score-window   (body = {"window": [[..d..], ..., W rows]})
    GET  /alerts/recent  (last N alerts)

Run with:
    uvicorn diploma_nids.inference.service:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..data import Preprocessor, WindowingConfig
from ..models import build_model, load_classical
from ..utils import get_logger, load_yaml, setup_logging
from .alerting import AlertFormer
from .stream import WindowScorer

logger = get_logger(__name__)
setup_logging("INFO")

app = FastAPI(title="diploma-nids", version="0.1.0")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class _State:
    preprocessor: Preprocessor | None = None
    model: Any = None
    window_cfg: WindowingConfig | None = None
    threshold: float = 0.5
    is_classical: bool = False
    scorer: WindowScorer | None = None
    alert_former: AlertFormer | None = None


state = _State()


def _load_state_from_env() -> None:
    """Load model and preprocessor based on environment variables."""
    model_yaml = os.environ.get("DIPLOMA_MODEL_YAML")
    ckpt = os.environ.get("DIPLOMA_CHECKPOINT")
    prep_path = os.environ.get("DIPLOMA_PREPROCESSOR")
    threshold = float(os.environ.get("DIPLOMA_THRESHOLD", "0.5"))

    if not (model_yaml and ckpt and prep_path):
        logger.warning("DIPLOMA_MODEL_YAML/DIPLOMA_CHECKPOINT/DIPLOMA_PREPROCESSOR not set; service runs in stub mode")
        return

    cfg = load_yaml(model_yaml)
    name = cfg["name"]
    state.is_classical = "architecture" not in cfg
    state.preprocessor = Preprocessor.load(prep_path)
    if state.is_classical:
        state.model = load_classical(ckpt)
    else:
        state.model = build_model(name, cfg)
        state.model.load(ckpt)
        ws = int(cfg["architecture"].get("window_size", 32))
        stride = max(1, ws // 4)
        state.window_cfg = WindowingConfig(window_size=ws, stride=stride)
        state.scorer = WindowScorer(state.preprocessor, state.model, state.window_cfg)

    state.threshold = threshold
    state.alert_former = AlertFormer(threshold=threshold)
    logger.info("loaded model=%s classical=%s threshold=%.3f", name, state.is_classical, threshold)


@app.on_event("startup")
def _startup() -> None:
    _load_state_from_env()


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------
class FlowRecordPayload(BaseModel):
    records: list[dict[str, Any]]


class WindowPayload(BaseModel):
    window: list[list[float]]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict[str, Any]:
    return {
        "loaded": state.model is not None,
        "is_classical": state.is_classical,
        "threshold": state.threshold,
        "expects_windows": (state.window_cfg is not None),
    }


@app.post("/score")
def score_records(payload: FlowRecordPayload) -> dict[str, Any]:
    if state.model is None or state.preprocessor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    df = pd.DataFrame(payload.records)
    df_pp = state.preprocessor.transform(df)
    feat_cols = state.preprocessor.state.output_columns
    X = df_pp[feat_cols].to_numpy(dtype=np.float32)

    if state.is_classical:
        scores = state.model.score(X)
    elif state.scorer is not None:
        scores = state.scorer.feed(df)
    else:
        # DL model on tabular
        import torch as _t
        with _t.no_grad():
            scores = state.model(_t.from_numpy(X)).probs.cpu().numpy()

    alerts: list[dict] = []
    if state.alert_former:
        for s in scores:
            a = state.alert_former.maybe_alert(float(s), attack_cat_pred=None, profile={})
            if a:
                alerts.append(a.__dict__)

    return {
        "n_inputs": int(len(df)),
        "n_scores": int(len(scores)),
        "scores": [float(s) for s in scores],
        "alerts": alerts,
    }


@app.post("/score-window")
def score_window(payload: WindowPayload) -> dict[str, Any]:
    if state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    import torch as _t
    arr = np.asarray(payload.window, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(status_code=400, detail="window must be 2D (W, F)")
    x = _t.from_numpy(arr[None, ...])
    with _t.no_grad():
        out = state.model(x)
    return {"score": float(out.probs.cpu().item())}


@app.get("/alerts/recent")
def recent_alerts(n: int = 50) -> dict[str, Any]:
    if not state.alert_former:
        return {"alerts": []}
    return {"alerts": [a.__dict__ for a in state.alert_former.recent(n)]}
