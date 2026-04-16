import hashlib
import json
import os
from contextlib import asynccontextmanager

import asyncpg
import numpy as np
import pandas as pd
import joblib
import redis.asyncio as aioredis
import sentry_sdk
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Sentry ────────────────────────────────────────────────────────────────────

SENTRY_DSN = os.getenv("SENTRY_DSN", "")
if SENTRY_DSN:
    sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=0.2)

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

# ── Config ────────────────────────────────────────────────────────────────────

ARTIFACT_DIR  = os.getenv("ARTIFACT_DIR", "../model/artifacts/")
DATABASE_URL  = os.getenv("DATABASE_URL", "postgresql://mlops:password123@localhost:5432/mlops")
REDIS_URL     = os.getenv("REDIS_URL",    "redis://localhost:6379/0")

models:   dict = {}
db_pool:  asyncpg.Pool | None = None
redis_client: aioredis.Redis | None = None

# ── DB schema ─────────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                    SERIAL PRIMARY KEY,
    est_diameter_min      FLOAT  NOT NULL,
    est_diameter_max      FLOAT  NOT NULL,
    relative_velocity     FLOAT  NOT NULL,
    absolute_magnitude    FLOAT  NOT NULL,
    miss_distance         FLOAT  NOT NULL,
    hazardous             BOOLEAN NOT NULL,
    hazardous_probability FLOAT   NOT NULL,
    miss_distance_km      FLOAT   NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client

    # Load ML artifacts
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

    # Connect to PostgreSQL
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        async with db_pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)
        print("PostgreSQL connected.")
    except Exception as exc:
        print(f"WARNING: PostgreSQL unavailable — {exc}")
        db_pool = None

    # Connect to Redis
    try:
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        print("Redis connected.")
    except Exception as exc:
        print(f"WARNING: Redis unavailable — {exc}")
        redis_client = None

    yield

    models.clear()
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.aclose()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "success", "message": "MLOps Server is Live!"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "category": "ML-Model"}

# ── Prediction endpoint ───────────────────────────────────────────────────────

def _cache_key(body: AsteroidInput) -> str:
    raw = f"{body.est_diameter_min}:{body.est_diameter_max}:{body.relative_velocity}:{body.absolute_magnitude}:{body.miss_distance}"
    return "pred:" + hashlib.sha256(raw.encode()).hexdigest()

@app.post("/api/predict", response_model=PredictionOutput)
async def predict(body: AsteroidInput):
    if "clf" not in models:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training notebooks first.")

    # Redis cache lookup
    cache_key = _cache_key(body)
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            return PredictionOutput(**json.loads(cached))

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

    result = PredictionOutput(
        hazardous=hazardous_label,
        hazardous_probability=round(hazardous_probability, 4),
        miss_distance_km=round(miss_distance_km, 2),
    )

    # Cache result in Redis (TTL: 1 hour)
    if redis_client:
        await redis_client.set(cache_key, result.model_dump_json(), ex=3600)

    # Log prediction to PostgreSQL
    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO predictions
                       (est_diameter_min, est_diameter_max, relative_velocity,
                        absolute_magnitude, miss_distance,
                        hazardous, hazardous_probability, miss_distance_km)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
                    body.est_diameter_min, body.est_diameter_max,
                    body.relative_velocity, body.absolute_magnitude,
                    body.miss_distance,
                    result.hazardous, result.hazardous_probability,
                    result.miss_distance_km,
                )
        except Exception as exc:
            print(f"WARNING: Failed to log prediction — {exc}")

    return result
