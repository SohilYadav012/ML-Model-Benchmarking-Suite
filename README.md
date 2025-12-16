# ML-Model-Benchmarking-Suite
A Python-based machine learning benchmarking suite that trains and compares multiple models (Logistic Regression, Random Forest, XGBoost, and Neural Networks) using accuracy, training time, and inference latency to identify the best model for deployment.


## ml_model_benchmarking

Production-quality benchmarking suite for comparing multiple machine learning
models on a common tabular dataset.

### Features

- **Real-world dataset**: Uses the Breast Cancer Wisconsin diagnostic dataset
  from scikit-learn (binary classification).
- **Preprocessing**: Missing value imputation, categorical encoding, and
  numerical feature scaling via a reusable sklearn pipeline.
- **Models**:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Neural network (sklearn MLP)
  - Neural network (TensorFlow/Keras)
- **Benchmarking metrics**:
  - Accuracy (primary metric)
  - Training time
  - Inference time
- **Failure analysis**:
  - Misclassified samples are logged per model to CSV files.
- **Optimisation step**:
  - Hyperparameter tuning for Random Forest via `GridSearchCV`.
- **Reporting**:
  - Tabular results (CSV and text)
  - Natural-language summary of model trade-offs.

### Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the full benchmarking pipeline:

```bash
python main.py
```

Outputs will be written to the `outputs/` directory:

- `benchmark_results.csv` – structured benchmark metrics
- `benchmark_results.txt` – human-readable table
- `summary.txt` – natural-language summary
- `outputs/failures/*.csv` – per-model failure cases

### Reproducibility

All models use a fixed random seed (`RANDOM_STATE = 42`) to ensure that results
are reproducible across runs, subject to the usual non-determinism present in
some deep learning operations.


