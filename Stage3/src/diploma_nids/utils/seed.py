"""Seed control for reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix all relevant RNG seeds.

    Affects: ``random``, ``numpy``, ``torch`` (CPU and CUDA), ``PYTHONHASHSEED``.
    When ``deterministic=True``, also enables PyTorch deterministic algorithms
    (``torch.use_deterministic_algorithms``) and disables cuDNN benchmarking.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding callback."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
