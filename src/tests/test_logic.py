# tests/test_logic.py
import numpy as np
import xgboost as xgb
import pytest
import optuna
from unittest.mock import MagicMock
from logic.trainer import objective
import mlflow
import matplotlib.pyplot as plt
import tempfile
import os

@pytest.fixture
def dummy_data():
    X = np.random.rand(20, 5)
    y = np.random.rand(20)

    dtrain = xgb.DMatrix(X[:10], label=y[:10])
    dvalid = xgb.DMatrix(X[10:15], label=y[10:15])
    dtest = xgb.DMatrix(X[15:], label=y[15:])

    return dtrain, dvalid, dtest, X, y


def test_objective_runs(dummy_data, monkeypatch):
    dtrain, dvalid, dtest, X, y = dummy_data

    # Mock globals inside the module
    monkeypatch.setattr("src.training_script.dtrain", dtrain)
    monkeypatch.setattr("src.training_script.dvalid", dvalid)
    monkeypatch.setattr("src.training_script.y_val", y[10:15])

    # Mock MLflow so it doesn't write files
    monkeypatch.setattr("mlflow.start_run", MagicMock())
    monkeypatch.setattr("mlflow.log_metric", MagicMock())
    monkeypatch.setattr("mlflow.log_params", MagicMock())

    trial = optuna.trial.FixedTrial({
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 1.0,
    })

    rmse = objective(trial)

    assert isinstance(rmse, float)
    assert rmse > 0


def test_optuna_study_runs(monkeypatch):
    # Prevent MLflow from writing anything
    monkeypatch.setattr("mlflow.start_run", MagicMock())
    monkeypatch.setattr("mlflow.log_metric", MagicMock())
    monkeypatch.setattr("mlflow.log_params", MagicMock())

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)

    assert len(study.trials) == 1
    assert study.best_value is not None


def test_mlflow_model_logging():
    # Use a temporary directory for MLflow
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    mlflow.set_tracking_uri(f"file://{tmp_dir}")

    # Create a temporary experiment
    experiment_id = mlflow.create_experiment("test_experiment")
    mlflow.set_experiment(experiment_id=experiment_id)

    X = np.random.rand(10, 5)
    y = np.random.rand(10)
    dtrain = xgb.DMatrix(X, label=y)

    params = {"objective": "reg:squarederror", "eval_metric": "rmse"}
    model = xgb.train(params, dtrain, num_boost_round=5)

    # Patch log_model to verify it is called
    from unittest.mock import patch
    with patch("mlflow.xgboost.log_model") as mock_log_model:
        mlflow.xgboost.log_model(model, artifact_path="model")
        mock_log_model.assert_called_once()



def test_feature_importance_plot():
    X = np.random.rand(10, 5)
    y = np.random.rand(10)
    dtrain = xgb.DMatrix(X, label=y)

    params = {"objective": "reg:squarederror", "eval_metric": "rmse"}
    model = xgb.train(params, dtrain, num_boost_round=5)

    fig, ax = plt.subplots()
    xgb.plot_importance(model, ax=ax)

    assert fig is not None

