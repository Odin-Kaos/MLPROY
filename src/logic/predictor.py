import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

classifier = xgb.XGBRegressor()
classifier.load_model(ROOT / "models" / "model.ubj")


def predict_class(vect: np.ndarray) -> bool:
    vect = np.array(vect, dtype=float).reshape(1, -1)
    pred = classifier.predict(vect)[0]
    return bool(pred > 0.5)
