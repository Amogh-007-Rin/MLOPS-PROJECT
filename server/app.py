from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import json

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class AsteroidInput(BaseModel):
    est_diameter_min:   float
    est_diameter_max:   float
    relative_velocity:  float
    absolute_magnitude: float
    miss_distance:      float

class PredictionOutput(BaseModel):
    hazardous:             bool
    hazardous_probability: float
    miss_distance_km:      float

# ── Model loading at startup ──────────────────────────────────────────────────

ARTIFACT_DIR = "../model/artifacts/"
models: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        models["clf"]          = joblib.load(f"{ARTIFACT_DIR}classifier.joblib")
        models["reg"]          = joblib.load(f"{ARTIFACT_DIR}regressor.joblib")
        models["scaler_clf"]   = joblib.load(f"{ARTIFACT_DIR}scaler_clf.joblib")
        models["scaler_reg"]   = joblib.load(f"{ARTIFACT_DIR}scaler_reg.joblib")
        with open(f"{ARTIFACT_DIR}feature_names.json") as f:
            models["feature_names"] = json.load(f)
        print("ML models loaded successfully.")
    except FileNotFoundError:
        print("WARNING: Model artifacts not found. Run the training notebooks first.")
    yield
    models.clear()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "success", "message": "MLOps Server is Live!"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "category": "ML-Model"}

# ── Prediction endpoint ───────────────────────────────────────────────────────

@app.post("/api/predict", response_model=PredictionOutput)
def predict(body: AsteroidInput):
    if "clf" not in models:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training notebooks first.")

    CLF_FEATURES = models["feature_names"]["clf_features"]
    REG_FEATURES = models["feature_names"]["reg_features"]

    # Feature engineering (mirrors Pre-processing.ipynb)
    diameter_avg          = (body.est_diameter_min + body.est_diameter_max) / 2
    diameter_ratio        = body.est_diameter_max / body.est_diameter_min
    log_diameter_avg      = np.log1p(diameter_avg)
    log_diameter_ratio    = np.log1p(diameter_ratio)
    log_relative_velocity = np.log1p(body.relative_velocity)
    log_miss_distance     = np.log1p(body.miss_distance)

    clf_input = pd.DataFrame([[
        log_diameter_avg, log_diameter_ratio, log_relative_velocity,
        log_miss_distance, body.absolute_magnitude, diameter_avg, diameter_ratio
    ]], columns=CLF_FEATURES)

    reg_input = pd.DataFrame([[
        log_diameter_avg, log_diameter_ratio, log_relative_velocity,
        body.absolute_magnitude, diameter_avg, diameter_ratio
    ]], columns=REG_FEATURES)

    clf_scaled = models["scaler_clf"].transform(clf_input)
    reg_scaled = models["scaler_reg"].transform(reg_input)

    hazardous_label       = bool(models["clf"].predict(clf_scaled)[0])
    hazardous_probability = float(models["clf"].predict_proba(clf_scaled)[0, 1])
    miss_distance_km      = float(np.expm1(models["reg"].predict(reg_scaled)[0]))

    return PredictionOutput(
        hazardous=hazardous_label,
        hazardous_probability=round(hazardous_probability, 4),
        miss_distance_km=round(miss_distance_km, 2),
    )
