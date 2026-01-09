"""
Model prediction utility.
"""

from pathlib import Path
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]

# Load trained model
classifier = xgb.XGBRegressor()
classifier.load_model(ROOT / "models" / "model.ubj")


def predict_class(vect: np.ndarray) -> bool:
    """
    Predict class for a given vector.

    Args:
        vect: Input features (1D array)

    Returns:
        True if predicted value > 0.5, else False
    """
    vect = np.array(vect, dtype=float).reshape(1, -1)
    pred = classifier.predict(vect)[0]
    return bool(pred > 0.5)

