"""
Evaluation utilities for the ml_model_benchmarking project.

This module provides functions to compute standard performance metrics as well
as to identify failure cases (misclassified or high-error samples).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


TaskType = Literal["classification", "regression"]


@dataclass
class EvaluationResult:
    """
    Container for evaluation metrics and artefacts.

    Attributes
    ----------
    primary_metric : float
        Accuracy for classification or RMSE for regression.
    additional_metrics : dict
        Optional additional metrics such as precision/recall/F1 per class.
    failures : pd.DataFrame
        DataFrame containing failure cases (misclassified/high-error samples).
    """

    primary_metric: float
    additional_metrics: Dict[str, float]
    failures: pd.DataFrame


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_original: pd.DataFrame,
    index: pd.Index,
) -> EvaluationResult:
    """
    Evaluate a classification model and identify misclassified samples.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    X_original : pd.DataFrame
        Original (preprocessed) feature matrix for reference in failures.
    index : pd.Index
        Index of the samples corresponding to predictions.

    Returns
    -------
    EvaluationResult
        Evaluation metrics and a DataFrame of misclassified samples.
    """
    acc = accuracy_score(y_true, y_pred)

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    # Flatten the nested dict for easier reporting
    metrics_flat: Dict[str, float] = {}
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                metrics_flat[f"{label}_{metric_name}"] = float(value)

    misclassified_mask = y_true != y_pred
    failures_df = X_original.loc[index[misclassified_mask]].copy()
    failures_df["true_label"] = y_true[misclassified_mask]
    failures_df["predicted_label"] = y_pred[misclassified_mask]

    return EvaluationResult(
        primary_metric=float(acc),
        additional_metrics=metrics_flat,
        failures=failures_df,
    )


__all__ = ["TaskType", "EvaluationResult", "evaluate_classification"]


