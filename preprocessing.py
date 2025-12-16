"""
Preprocessing utilities for the ml_model_benchmarking project.

This module provides a reusable preprocessing pipeline that handles:

- Missing value imputation
- Categorical feature encoding
- Numerical feature scaling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing.

    Attributes
    ----------
    numerical_strategy : str
        Imputation strategy for numerical features.
    categorical_strategy : str
        Imputation strategy for categorical features.
    scale_numerical : bool
        Whether to apply standard scaling to numerical features.
    """

    numerical_strategy: str = "median"
    categorical_strategy: str = "most_frequent"
    scale_numerical: bool = True


def build_preprocessing_pipeline(
    X: pd.DataFrame, config: PreprocessingConfig
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a preprocessing pipeline for the given dataset.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    config : PreprocessingConfig
        Configuration specifying preprocessing behaviour.

    Returns
    -------
    tuple
        A tuple ``(preprocessor, numerical_features, categorical_features)`` where:

        - ``preprocessor`` is a ``ColumnTransformer`` that can be used inside
          an sklearn ``Pipeline``.
        - ``numerical_features`` is a list of numerical feature names.
        - ``categorical_features`` is a list of categorical feature names.
    """
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Numerical pipeline
    numeric_transformers = [("imputer", SimpleImputer(strategy=config.numerical_strategy))]
    if config.scale_numerical:
        numeric_transformers.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_transformers)

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=config.categorical_strategy, fill_value="missing")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numerical_features, categorical_features


__all__ = ["PreprocessingConfig", "build_preprocessing_pipeline"]


