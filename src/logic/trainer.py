import optuna
from optuna.integration import MLflowCallback
import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from logic.load_data import loadCSV

ROOT = Path(__file__).resolve().parents[2]
tracking_dir = ROOT / "logs"

X_train, y_train, X_val, y_val, X_test, y_test = loadCSV()

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

mlflow_callback = MLflowCallback(
    tracking_uri=f"file:{tracking_dir.as_posix()}", metric_name="rmse"
)


def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "eta": trial.suggest_float("eta", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
    }
    with mlflow.start_run(nested=True):
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        preds = model.predict(dvalid)
        rmse = mean_squared_error(y_val, preds) ** 0.5

        mlflow.log_metric("rmse", rmse)
        mlflow.log_params(params)

        return rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5, callbacks=[mlflow_callback])

print("Best params:", study.best_params)
print("Best RMSE:", study.best_value)

best_params = study.best_params
best_params["objective"] = "reg:squarederror"
best_params["eval_metric"] = "rmse"

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_val, label=y_val)
with mlflow.start_run(run_name="best_model"):
    best_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=500,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=50,
    )

    mlflow.xgboost.log_model(best_model, artifact_path="model")

best_model.save_model(ROOT / "models" / "model.ubj")

fig, ax = plt.subplots(figsize=(8, 6))
xgb.plot_importance(best_model, ax=ax)
plt.tight_layout()
fig.savefig(tracking_dir / "imgs/feature_importance.png")
mlflow.log_artifact(tracking_dir / "imgs/feature_importance.png")
