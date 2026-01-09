import mlflow
from pathlib import Path

# Force local MLflow tracking for tests & CI
TRACKING_DIR = Path(__file__).resolve().parents[1] / "logs"
mlflow.set_tracking_uri(f"file:{TRACKING_DIR.as_posix()}")

