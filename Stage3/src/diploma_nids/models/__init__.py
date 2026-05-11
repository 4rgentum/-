"""Models subpackage.

Importing this module triggers registration of all detectors in the registry
defined by ``base.py``. The proposed model is ``cnn_lstm``.
"""

from .autoencoder import VAE, Autoencoder  # noqa: F401
from .base import (  # noqa: F401
    BaseDetector,
    DLDetector,
    ModelOutput,
    build_model,
    get_model,
    list_models,
    register,
)
from .classical import (  # noqa: F401
    IsolationForestDetector,
    LogisticRegressionDetector,
    OneClassSVMDetector,
    RandomForestDetector,
    XGBoostDetector,
    load_classical,
    save_classical,
)
from .cnn1d import CNN1D  # noqa: F401
from .cnn_lstm import CNNLSTM  # noqa: F401
from .mlp import MLP  # noqa: F401
from .rnn_family import BiLSTMAttn, GRUDetector, LSTMDetector  # noqa: F401
from .tcn import TCN  # noqa: F401
from .transformer import TransformerEncoder  # noqa: F401

__all__ = [
    "BaseDetector",
    "DLDetector",
    "ModelOutput",
    "build_model",
    "get_model",
    "list_models",
    "register",
    "MLP",
    "CNN1D",
    "CNNLSTM",
    "LSTMDetector",
    "GRUDetector",
    "BiLSTMAttn",
    "TCN",
    "TransformerEncoder",
    "Autoencoder",
    "VAE",
    "LogisticRegressionDetector",
    "RandomForestDetector",
    "XGBoostDetector",
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "save_classical",
    "load_classical",
]
