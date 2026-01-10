"""
Serialization script: load the best model from MLflow and save it locally.
"""

import os
from pathlib import Path
import mlflow
import xgboost as xgb

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))

def get_best_run(experiment_id: str):
    """
    Returns the MLflow run with the lowest RMSE.
    """
    client = mlflow.tracking.MlflowClient()

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.rmse ASC"],  # lowest RMSE first
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No MLflow runs found. Train the model first.")

    return runs[0]


def load_best_model(run):
    """
    Loads the XGBoost model artifact from the best MLflow run.
    """
    artifact_uri = run.info.artifact_uri
    model_path = f"{artifact_uri}/model"

    # MLflow automatically detects XGBoost format
    model = mlflow.xgboost.load_model(model_path)
    return model


def save_model_locally(model):
    """
    Saves the model to models/model.ubj.
    """
    root = Path(__file__).resolve().parents[2]
    output_path = root / "models" / "model.ubj"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(output_path)
    print(f"Model saved to: {output_path}")


def main():
    # Configure MLflow tracking URI
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))

    # Get default experiment (ID = "0")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Default")

    if experiment is None:
        raise RuntimeError("MLflow experiment 'Default' not found.")

    best_run = get_best_run(experiment.experiment_id)
    print(f"Best run ID: {best_run.info.run_id}")
    print(f"Best RMSE: {best_run.data.metrics.get('rmse')}")

    model = load_best_model(best_run)
    save_model_locally(model)


if __name__ == "__main__":
    main()
