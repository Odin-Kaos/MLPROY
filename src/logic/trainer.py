"""
Training logic for XGBoost + Optuna.
"""
import os
import mlflow
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import optuna

# These globals are injected / monkeypatched by training_script or tests
dtrain = None
dvalid = None
y_val = None

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))

def objective(trial):
    """Optuna objective function for XGBoost regression."""
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
