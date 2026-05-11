"""Online window scoring for streamed flow records."""
from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from ..data import Preprocessor, WindowingConfig
from ..models.base import DLDetector


class WindowScorer:
    """Maintain a rolling buffer of preprocessed records and score windows."""

    def __init__(
        self,
        preprocessor: Preprocessor,
        model: DLDetector,
        window_cfg: WindowingConfig,
        device: str = "cpu",
    ) -> None:
        self.prep = preprocessor
        self.model = model.to(device)
        self.model.eval()
        self.window = int(window_cfg.window_size)
        self.stride = int(window_cfg.stride)
        self.device = device
        self.buffer: deque[np.ndarray] = deque(maxlen=self.window * 4)
        self._since_last = 0

    def feed(self, df_records: pd.DataFrame) -> list[float]:
        """Append records, return list of scores for windows that closed.

        Each score corresponds to a closed window; multiple windows may
        close per call if many records were appended at once.
        """
        df_pp = self.prep.transform(df_records)
        feat_cols = self.prep.state.output_columns
        arr = df_pp[feat_cols].to_numpy(dtype=np.float32)
        for row in arr:
            self.buffer.append(row)
        return self._drain()

    def _drain(self) -> list[float]:
        scores: list[float] = []
        while len(self.buffer) >= self.window:
            data = np.stack(list(self.buffer)[-self.window :], axis=0)
            x = torch.from_numpy(data[None, ...]).to(self.device)
            with torch.no_grad():
                out = self.model(x)
            scores.append(float(out.probs.cpu().item() if out.probs.ndim == 1 else out.probs.cpu().numpy()[0]))
            # Slide: drop ``stride`` oldest entries
            for _ in range(self.stride):
                if not self.buffer:
                    break
                self.buffer.popleft()
        return scores
