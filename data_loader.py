"""
Data loading utilities for the ml_model_benchmarking project.

This module is responsible for loading a real-world tabular dataset and
returning it as Pandas DataFrames suitable for downstream preprocessing
and modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


RANDOM_STATE: int = 42


@dataclass
class Dataset:
    """
    Container for tabular dataset components.

    Attributes
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    target_name : str
        Name of the target variable.
    feature_names : list[str]
        List of feature names.
    task_type : str
        Either ``\"classification\"`` or ``\"regression\"``.
    """

    X: pd.DataFrame
    y: pd.Series
    target_name: str
    feature_names: list[str]
    task_type: str


def load_tabular_dataset() -> Dataset:
    """
    Load a real-world classification dataset as a `Dataset`.

    Currently uses the Breast Cancer Wisconsin dataset from scikit-learn,
    which is a common real-world diagnostic dataset.

    Returns
    -------
    Dataset
        A dataset object containing features, target, and metadata.
    """
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    return Dataset(
        X=X,
        y=y,
        target_name=data.target_names[1]
        if isinstance(data.target_names, np.ndarray)
        else "target",
        feature_names=list(X.columns),
        task_type="classification",
    )


__all__ = ["Dataset", "load_tabular_dataset", "RANDOM_STATE"]


