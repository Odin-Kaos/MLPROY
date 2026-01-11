"""
FastAPI endpoints for model prediction, health check, and Prometheus metrics.
"""

from pathlib import Path
import time
import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.logic.predictor import predict_class

app = FastAPI()

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = Jinja2Templates(directory=str(ROOT / "templates"))

# ---------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------

# Count how many predictions have been made
PREDICTION_COUNT = Counter(
    "prediction_total",
    "Total number of predictions served"
)

# Track prediction latency
PREDICTION_LATENCY = Gauge(
    "prediction_latency_seconds",
    "Time spent processing a prediction"
)

# Track API health status (1 = healthy, 0 = unhealthy)
HEALTH_STATUS = Gauge(
    "api_health_status",
    "Health status of the API (1 = ok)"
)
HEALTH_STATUS.set(1)  # API starts healthy


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    """Render the homepage."""
    return TEMPLATES.TemplateResponse("home.html", {"request": request})


@app.post("/predict")
async def predict(vect: str = Form(...)):
    """
    Predict class for input vector.
    Expects a CSV-like string, e.g., "1,0,1,0,0,1".
    """
    start = time.time()

    vect_list = [float(x) for x in vect.split(",")]
    predicted_class = predict_class(np.array(vect_list))

    # Update Prometheus metrics
    PREDICTION_COUNT.inc()
    PREDICTION_LATENCY.set(time.time() - start)

    return {"predicted_class": predicted_class}


@app.get("/health")
async def health():
    """Health check endpoint."""
    HEALTH_STATUS.set(1)
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
