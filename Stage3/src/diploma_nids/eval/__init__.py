"""Evaluation subpackage."""

from .drift import (  # noqa: F401
    drift_report,
    kl_divergence,
    maximum_mean_discrepancy,
    population_stability_index,
)
from .error_analysis import confusion_matrix_dataframe, per_class_breakdown  # noqa: F401
from .metrics import (  # noqa: F401
    ClassificationReport,
    bootstrap_ci,
    classification_metrics,
    expected_calibration_error,
    find_threshold_for_target_fpr,
    find_threshold_max_f1,
    fp_per_time,
    integral_metrics,
    time_to_detect,
)
from .thresholding import TemperatureScaler, calibrate  # noqa: F401

__all__ = [
    "ClassificationReport",
    "classification_metrics",
    "integral_metrics",
    "expected_calibration_error",
    "fp_per_time",
    "time_to_detect",
    "find_threshold_for_target_fpr",
    "find_threshold_max_f1",
    "bootstrap_ci",
    "TemperatureScaler",
    "calibrate",
    "population_stability_index",
    "kl_divergence",
    "maximum_mean_discrepancy",
    "drift_report",
    "confusion_matrix_dataframe",
    "per_class_breakdown",
]
