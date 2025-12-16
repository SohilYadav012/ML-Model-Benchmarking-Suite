"""
Benchmarking utilities for the ml_model_benchmarking project.

This module is responsible for:
- Training and evaluating multiple models on the same dataset
- Measuring training and inference time
- Returning structured benchmark results
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

from data_loader import Dataset, RANDOM_STATE
from evaluation import EvaluationResult, evaluate_classification


@dataclass
class BenchmarkResult:
    """
    Benchmark results for a single model.

    Attributes
    ----------
    model_name : str
        Name of the model.
    primary_metric : float
        Accuracy (classification) or RMSE (regression).
    training_time : float
        Training time in seconds.
    inference_time : float
        Inference time in seconds for the test set.
    evaluation : EvaluationResult
        Detailed evaluation result including failures.
    extra_info : dict
        Optional extra information (e.g. number of parameters).
    """

    model_name: str
    primary_metric: float
    training_time: float
    inference_time: float
    evaluation: EvaluationResult
    extra_info: Dict[str, Any]


def _build_keras_model(input_dim: int) -> keras.Model:
    """
    Build a simple feed-forward neural network for binary classification.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def benchmark_sklearn_models(
    dataset: Dataset,
    models: Dict[str, Any],
    test_size: float = 0.2,
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark a collection of sklearn-style models.

    Parameters
    ----------
    dataset : Dataset
        Loaded dataset to benchmark on.
    models : dict
        Mapping from model name to sklearn-like estimator or pipeline.
    test_size : float
        Fraction of the dataset to use for testing.

    Returns
    -------
    dict
        Mapping from model name to ``BenchmarkResult``.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=dataset.y,
    )

    results: Dict[str, BenchmarkResult] = {}

    for name, model in models.items():
        start_train = time.perf_counter()
        model.fit(X_train, y_train)
        end_train = time.perf_counter()

        start_infer = time.perf_counter()
        y_pred = model.predict(X_test)
        end_infer = time.perf_counter()

        eval_res = evaluate_classification(
            y_true=y_test.to_numpy(),
            y_pred=np.asarray(y_pred),
            X_original=dataset.X.loc[X_test.index],
            index=X_test.index,
        )

        results[name] = BenchmarkResult(
            model_name=name,
            primary_metric=eval_res.primary_metric,
            training_time=end_train - start_train,
            inference_time=end_infer - start_infer,
            evaluation=eval_res,
            extra_info={},
        )

    return results


def benchmark_keras_model(
    dataset: Dataset,
    test_size: float = 0.2,
    epochs: int = 30,
    batch_size: int = 32,
) -> BenchmarkResult:
    """
    Benchmark a TensorFlow/Keras neural network on the dataset.

    Preprocessing is performed using standardization of numerical features.

    Parameters
    ----------
    dataset : Dataset
        Loaded dataset to benchmark on.
    test_size : float
        Fraction of the dataset to use for testing.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.

    Returns
    -------
    BenchmarkResult
        Benchmark result for the Keras model.
    """
    # For Keras, we work directly on numeric numpy arrays.
    X = dataset.X.to_numpy(dtype=np.float32)
    y = dataset.y.to_numpy().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=dataset.y,
    )

    # Simple feature scaling for Keras model
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    keras.utils.set_random_seed(RANDOM_STATE)

    model = _build_keras_model(input_dim=X_train_std.shape[1])

    start_train = time.perf_counter()
    model.fit(
        X_train_std,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    end_train = time.perf_counter()

    start_infer = time.perf_counter()
    y_pred_prob = model.predict(X_test_std, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    end_infer = time.perf_counter()

    # Reconstruct a DataFrame for failures using the original index
    X_original = pd.DataFrame(X_test, columns=dataset.feature_names)
    index = pd.RangeIndex(len(X_original))

    eval_res = evaluate_classification(
        y_true=y_test.astype(int),
        y_pred=y_pred,
        X_original=X_original,
        index=index,
    )

    num_params = model.count_params()

    return BenchmarkResult(
        model_name="keras_mlp",
        primary_metric=eval_res.primary_metric,
        training_time=end_train - start_train,
        inference_time=end_infer - start_infer,
        evaluation=eval_res,
        extra_info={"num_parameters": int(num_params)},
    )


__all__ = ["BenchmarkResult", "benchmark_sklearn_models", "benchmark_keras_model"]


