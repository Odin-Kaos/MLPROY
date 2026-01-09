"""
FastAPI endpoints for model prediction and health check.
"""

from pathlib import Path
import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from logic.predictor import predict_class

app = FastAPI()

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = Jinja2Templates(directory=str(ROOT / "templates"))


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
    vect_list = [float(x) for x in vect.split(",")]
    predicted_class = predict_class(np.array(vect_list))
    return {"predicted_class": predicted_class}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

