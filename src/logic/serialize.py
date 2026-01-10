"""
Serialization script: load best parameters from MLflow logs in /logs,
retrain the model using those parameters, and save it into /models/model.ubj.
"""

import os
from pathlib import Path
import mlflow
import xgboost as xgb
from load_data import load_csv


# -----------------------------
# Paths and MLflow setup
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]
tracking_dir = ROOT / "logs"
models_dir = ROOT / "models"
models_dir.mkdir(exist_ok=True)

# Use environment variable if provided (CI/CD), otherwise default to /logs
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"file:{tracking_dir.as_posix()}"))


# -----------------------------
# Load data
# -----------------------------
X_train, y_train, X_val, y_val, X_test, y_test = load_csv()

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_val, label=y_val)


# -----------------------------
# Helper: find best run
# -----------------------------
def get_best_run():
    client = mlflow.tracking.MlflowClient()

    # Default experiment is usually "0"
    experiment = client.get_experiment_by_name("Default")
    if experiment is None:
        raise RuntimeError("MLflow experiment 'Default' not found in /logs")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No MLflow runs found in /logs. Train the model first.")

    return runs[0]


# -----------------------------
# Main logic
# -----------------------------
def main():
    print("Searching for best run in MLflow logs...")

    best_run = get_best_run()
    best_params = best_run.data.params

    print("Best run ID:", best_run.info.run_id)
    print("Best RMSE:", best_run.data.metrics.get("rmse"))
    print("Best parameters:", best_params)

    # Convert params from strings to correct types
    final_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": int(best_params["max_depth"]),
        "eta": float(best_params["eta"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "lambda": float(best_params["lambda"]),
        "alpha": float(best_params["alpha"]),
    }

    print("Retraining model with best parameters...")

    with mlflow.start_run(run_name="serialization_retrain"):
        model = xgb.train(
            final_params,
            dtrain,
            num_boost_round=500,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=50,
        )

        # Save model to MLflow
        mlflow.xgboost.log_model(model, artifact_path="model")

    # Save model locally
    output_path = models_dir / "model.ubj"
    model.save_model(output_path)

    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
