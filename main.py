"""
Entry point for the ml_model_benchmarking project.

Running this script will:

1. Load a real-world tabular dataset.
2. Build preprocessing and model pipelines.
3. Benchmark multiple models (including a Keras neural network).
4. Identify and log failure cases for each model.
5. Apply a simple optimisation step (Random Forest hyperparameter tuning).
6. Re-run the benchmark for the optimised model.
7. Print a tabular report and a natural-language summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from benchmark import BenchmarkResult, benchmark_keras_model, benchmark_sklearn_models
from data_loader import Dataset, RANDOM_STATE, load_tabular_dataset
from models import ModelConfig, build_models
from preprocessing import PreprocessingConfig, build_preprocessing_pipeline
from report import build_results_table, generate_summary
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def _ensure_output_dir() -> Path:
    """Ensure the output directory for logs and reports exists."""
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def log_failures(
    results: Dict[str, BenchmarkResult],
    output_dir: Path,
) -> None:
    """
    Persist failure cases for each model into CSV files.

    Parameters
    ----------
    results : dict
        Mapping from model name to benchmark result.
    output_dir : Path
        Directory where failure CSV files will be saved.
    """
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)

    for name, res in results.items():
        failures: pd.DataFrame = res.evaluation.failures
        if failures.empty:
            continue
        failures_path = failures_dir / f"{name}_failures.csv"
        failures.to_csv(failures_path, index=False)


def optimise_random_forest(
    dataset: Dataset,
    preprocessor,
) -> Pipeline:
    """
    Apply a simple hyperparameter optimisation for the Random Forest model.

    A small grid search is performed to keep runtime reasonable.

    Parameters
    ----------
    dataset : Dataset
        Dataset to fit on.
    preprocessor :
        Preprocessing transformer to include in the pipeline.

    Returns
    -------
    Pipeline
        Optimised Random Forest pipeline.
    """
    from sklearn.model_selection import train_test_split

    X_train, _, y_train, _ = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataset.y,
    )

    base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", base_rf)])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="accuracy",
    )
    search.fit(X_train, y_train)
    print("Best Random Forest parameters:", search.best_params_)

    return search.best_estimator_


def main() -> None:
    """Run the full benchmarking pipeline."""
    output_dir = _ensure_output_dir()

    # 1. Load dataset
    dataset = load_tabular_dataset()

    # 2. Build preprocessing
    preproc_config = PreprocessingConfig()
    preprocessor, _, _ = build_preprocessing_pipeline(dataset.X, preproc_config)

    # 3. Build models
    model_config = ModelConfig(task_type=dataset.task_type, random_state=RANDOM_STATE)
    sklearn_models = build_models(preprocessor, model_config)

    # 4. Benchmark sklearn models
    sklearn_results = benchmark_sklearn_models(dataset, sklearn_models)


    # 5. Benchmark Keras model
    keras_result = benchmark_keras_model(dataset)
    all_results: Dict[str, BenchmarkResult] = dict(sklearn_results)
    all_results[keras_result.model_name] = keras_result

    # Log failures for baseline models
    log_failures(all_results, output_dir)

    # 6. Optimise Random Forest and re-benchmark
    optimised_rf = optimise_random_forest(dataset, preprocessor)
    optimised_models = {"random_forest_optimised": optimised_rf}
    optimised_results = benchmark_sklearn_models(dataset, optimised_models)

    # Merge optimised result
    all_results.update(optimised_results)

    # 7. Reporting
    results_table = build_results_table(all_results)
    summary_text = generate_summary(all_results)

    # Save report artifacts
    results_table.to_csv(output_dir / "benchmark_results.csv", index=False)
    (output_dir / "benchmark_results.txt").write_text(results_table.to_string(index=False))
    (output_dir / "summary.txt").write_text(summary_text)

    # Print to stdout for convenience
    print("\n=== Benchmark Results ===")
    print(results_table.to_string(index=False))
    print("\n=== Summary ===")
    print(summary_text)


if __name__ == "__main__":
    main()


