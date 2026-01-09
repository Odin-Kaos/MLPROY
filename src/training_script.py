"""
Training script wrapper.

- Loads data
- Exposes globals expected by tests
- Injects globals into logic.trainer
- Runs Optuna + MLflow
"""

from pathlib import Path
import optuna
import xgboost as xgb
from optuna.integration import MLflowCallback
import mlflow

from logic.load_data import load_csv
import logic.trainer as trainer

# Paths
ROOT = Path(__file__).resolve().parents[1]
TRACKING_DIR = ROOT / "logs"

# Force safe local MLflow tracking
mlflow.set_tracking_uri(f"file:{TRACKING_DIR.as_posix()}")

# Load dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_csv()

# Globals required by tests
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Inject into trainer module
trainer.dtrain = dtrain
trainer.dvalid = dvalid
trainer.y_val = y_val

# MLflow callback for Optuna
mlflow_callback = MLflowCallback(
    tracking_uri=f"file:{TRACKING_DIR.as_posix()}",
    metric_name="rmse",
)


def main():
    """Run an Optuna study and log metrics to MLflow."""
    study = optuna.create_study(direction="minimize")
    study.optimize(trainer.objective, n_trials=5, callbacks=[mlflow_callback])

    print("Best params:", study.best_params)
    print("Best RMSE:", study.best_value)


if __name__ == "__main__":
    main()

