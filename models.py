"""
Model construction utilities for the ml_model_benchmarking project.

This module defines a common interface for training and evaluating multiple
machine learning models on the same preprocessed dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


TaskType = Literal["classification", "regression"]


@dataclass
class ModelConfig:
    """
    Configuration for the family of models used in benchmarking.

    Attributes
    ----------
    task_type : TaskType
        Either ``\"classification\"`` or ``\"regression\"``. Currently only
        classification is supported.
    random_state : int
        Random seed for reproducibility.
    """

    task_type: TaskType = "classification"
    random_state: int = 42


def build_models(preprocessor, config: ModelConfig) -> Dict[str, Pipeline]:
    """
    Build a dictionary of model pipelines keyed by a human-readable name.

    Parameters
    ----------
    preprocessor :
        A fitted or unfitted sklearn-compatible transformer (typically a
        ``ColumnTransformer``) used for preprocessing.
    config : ModelConfig
        Configuration specifying global modeling options.

    Returns
    -------
    dict
        Mapping from model name to an sklearn ``Pipeline`` instance.
    """
    if config.task_type != "classification":
        raise NotImplementedError("Only classification task is implemented in this project.")

    models: Dict[str, Pipeline] = {}

    logistic_regression = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        random_state=config.random_state,
    )

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=config.random_state,
    )

    xgb_classifier = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    # Simple neural network using sklearn's MLPClassifier; a TensorFlow/Keras
    # model is used separately in `benchmark.py` to collect deep-learning
    # specific metrics.
    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=config.random_state,
    )

    models["logistic_regression"] = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", logistic_regression)]
    )
    models["random_forest"] = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", random_forest)]
    )
    models["xgboost"] = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", xgb_classifier)]
    )
    models["mlp_sklearn"] = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", mlp_classifier)]
    )

    return models


__all__ = ["ModelConfig", "TaskType", "build_models"]


