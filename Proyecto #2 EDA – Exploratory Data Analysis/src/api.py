# src/api.py
from typing import Optional
import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .inference import predict


app = FastAPI(title="Store Sales Forecast API", version="0.1.0")


class PredictionRequest(BaseModel):
    store_nbr: int
    family: str
    date: str  # "YYYY-MM-DD"
    onpromotion: Optional[int] = 0


@app.get("/")
def root():
    return {"status": "ok"}


from datetime import datetime, timezone

@app.post("/predict")
def make_prediction(data: PredictionRequest):
    preds = predict(data)  
    now = datetime.now(timezone.utc).isoformat()

    return {
        "predictions": [float(preds[0])],
        "model_metrics": load_metrics(),   
        "model_params": load_params(),    
        "now": now
    }


