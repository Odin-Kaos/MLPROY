# src/api/api.py
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from logic.predictor import predict_class
import numpy as np
from pathlib import Path

app = FastAPI()

ROOT = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(ROOT / "templates"))


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict")
async def predict(vect: str = Form(...)):
    # Expecting something like: "1,0,1,0,0,1"
    vect_list = [float(x) for x in vect.split(",")]
    pred_class = predict_class(np.array(vect_list))

    return {"predicted_class": pred_class}


@app.get("/health")
async def health():
    return {"status": "ok"}
