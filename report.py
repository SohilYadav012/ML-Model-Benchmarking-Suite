"""
Reporting utilities for the ml_model_benchmarking project.

This module converts structured benchmark results into human-readable tables
and natural-language summaries describing model trade-offs.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from benchmark import BenchmarkResult


def build_results_table(results: Dict[str, BenchmarkResult]) -> pd.DataFrame:
    """
    Build a Pandas DataFrame summarising benchmark results.

    Parameters
    ----------
    results : dict
        Mapping from model name to ``BenchmarkResult``.

    Returns
    -------
    pd.DataFrame
        Table with one row per model including metric and timing columns.
    """
    rows: List[dict] = []
    for name, res in results.items():
        row = {
            "model": name,
            "primary_metric": res.primary_metric,
            "training_time_sec": res.training_time,
            "inference_time_sec": res.inference_time,
        }
        row.update({f"extra_{k}": v for k, v in res.extra_info.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by="primary_metric", ascending=False).reset_index(drop=True)
    return df


def generate_summary(results: Dict[str, BenchmarkResult]) -> str:
    """
    Generate a natural-language summary of benchmark results.

    The summary highlights best-performing models, trade-offs between accuracy
    and latency, and any noteworthy resource characteristics.

    Parameters
    ----------
    results : dict
        Mapping from model name to ``BenchmarkResult``.

    Returns
    -------
    str
        Human-readable summary of model performance.
    """
    if not results:
        return "No benchmark results available."

    table = build_results_table(results)

    best_row = table.iloc[0]
    best_model = best_row["model"]
    best_metric = best_row["primary_metric"]

    fastest_train = table.sort_values("training_time_sec").iloc[0]
    fastest_infer = table.sort_values("inference_time_sec").iloc[0]

    lines: List[str] = []
    lines.append(
        f"The best model in terms of primary metric (accuracy) is '{best_model}' "
        f"with an accuracy of {best_metric:.4f}."
    )
    if fastest_train["model"] == best_model:
        lines.append(
            f"'{best_model}' is also the fastest to train "
            f"({fastest_train['training_time_sec']:.4f} seconds), making it a strong overall choice."
        )
    else:
        lines.append(
            f"The fastest model to train is '{fastest_train['model']}' "
            f"({fastest_train['training_time_sec']:.4f} seconds), "
            f"which trades some accuracy ({fastest_train['primary_metric']:.4f}) for speed."
        )

    if fastest_infer["model"] == best_model:
        lines.append(
            f"'{best_model}' also has the lowest inference latency "
            f"({fastest_infer['inference_time_sec']:.6f} seconds on the test set)."
        )
    else:
        lines.append(
            f"The lowest inference latency is achieved by '{fastest_infer['model']}' "
            f"({fastest_infer['inference_time_sec']:.6f} seconds), which is well-suited "
            "for real-time scenarios."
        )

    if "keras_mlp" in results:
        keras_res = results["keras_mlp"]
        num_params = keras_res.extra_info.get("num_parameters")
        if num_params is not None:
            lines.append(
                f"The Keras MLP has approximately {num_params:,} trainable parameters, "
                "which may require more resources but can capture complex non-linear patterns."
            )

    lines.append(
        "Overall, choose the model that best matches your deployment constraints: "
        "tree-based models like Random Forest and XGBoost often provide strong accuracy "
        "with robust performance, while linear models and small neural networks can offer "
        "lower latency and simpler decision boundaries."
    )

    return "\n".join(lines)


__all__ = ["build_results_table", "generate_summary"]


