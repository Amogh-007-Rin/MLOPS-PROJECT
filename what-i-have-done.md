# What Has Been Done — MLOPS-PROJECT

**Last updated:** 2026-03-26
**Project:** NASA Near-Earth Object Hazard Prediction — Full-Stack MLOps Application

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Full Project File Structure](#4-full-project-file-structure)
5. [Dataset](#5-dataset)
6. [What Was Built Before This Session (Git History)](#6-what-was-built-before-this-session-git-history)
7. [What Was Done in This Session (Claude)](#7-what-was-done-in-this-session-claude)
8. [Component Status](#8-component-status)
9. [ML Pipeline — Detailed Walkthrough](#9-ml-pipeline--detailed-walkthrough)
10. [API Contract](#10-api-contract)
11. [Infrastructure and Deployment](#11-infrastructure-and-deployment)
12. [How to Run the Project](#12-how-to-run-the-project)
13. [What Still Needs to Be Done](#13-what-still-needs-to-be-done)
14. [Key Design Decisions and Rationale](#14-key-design-decisions-and-rationale)

---

## 1. Project Overview

This is an end-to-end MLOps application that trains supervised machine learning models on NASA's Near-Earth Object (NEO) dataset and serves predictions through a REST API. The system simultaneously solves two ML tasks:

- **Classification**: Given an asteroid's physical characteristics, predict whether it is a Potentially Hazardous Object (PHO) — `True` or `False`
- **Regression**: Given an asteroid's physical characteristics, predict its closest approach distance to Earth in km

The predictions are served through a FastAPI backend (`POST /api/predict`) that loads the trained models at startup and responds in milliseconds.

The broader application stack also includes a React frontend, an Express.js worker service, a Rust WebSocket layer (placeholder), PostgreSQL, and Redis — all orchestrated through Docker Compose with Nginx as the gateway.

---

## 2. Architecture

```
Internet
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Nginx (port 80)  ─── Reverse Proxy Gateway             │
│    /         → React Client  (port 5173)                │
│    /api      → FastAPI Server (port 8000)               │
│    /docs     → FastAPI Swagger UI                       │
└─────────────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
  ┌─────────────┐     ┌───────────────────────┐
  │   React +   │     │  FastAPI (Python 3.11) │
  │ TypeScript  │     │  POST /api/predict     │
  │  (Vite)     │     │  loads ML models at    │
  └─────────────┘     │  startup via lifespan  │
                      └────────┬──────────────┘
                               │ reads
                               ▼
                      ┌───────────────────┐
                      │  model/artifacts/ │
                      │  classifier.joblib│
                      │  regressor.joblib │
                      │  scaler_clf.joblib│
                      │  scaler_reg.joblib│
                      │  feature_names.json│
                      │  model_metadata.json│
                      └───────────────────┘

Additional services (configured but not yet integrated):
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  Workers-SB  │   │  PostgreSQL  │   │    Redis     │
  │  (Express.js)│   │  (port 5432) │   │  (port 6379) │
  └──────────────┘   └──────────────┘   └──────────────┘

  ┌──────────────┐
  │  Rust Sockets│  ← placeholder only, not implemented
  └──────────────┘
```

---

## 3. Tech Stack

| Layer | Technology | Version | Status |
|---|---|---|---|
| **Frontend** | React + TypeScript | React 19.2, TS 5.9 | Scaffold only |
| **Frontend Build** | Vite | 7.3.1 | Configured |
| **Primary Backend** | FastAPI (Python) | 0.135.1 | Active — has predict endpoint |
| **ASGI Server** | Uvicorn | 0.41.0 | Configured |
| **Data Validation** | Pydantic | 2.12.5 | Used in API schemas |
| **Worker Service** | Express.js + TypeScript | Express 5.2, TS 5.9 | Health check only |
| **WebSocket Layer** | Rust | edition 2024 | Placeholder only |
| **Gateway** | Nginx | Alpine | Configured, routes `/` and `/api` |
| **Database** | PostgreSQL | 16 Alpine | In Docker, not wired to app |
| **Cache** | Redis | 7 Alpine | In Docker, not wired to app |
| **Containerisation** | Docker + Docker Compose | — | Fully configured |
| **CI/CD** | GitHub Actions | — | Configured but commented out |
| **ML Training** | scikit-learn | 1.6.1 | Added |
| **Hyperparameter Tuning** | Optuna | 4.2.1 | Added |
| **Class Imbalance** | imbalanced-learn | 0.13.0 | Added |
| **Data Analysis** | pandas + numpy | 3.0.1 / 2.4.3 | Present |
| **Visualisation** | matplotlib + seaborn | 3.10.1 / 0.13.2 | Added |
| **Model Serialisation** | joblib | 1.4.2 | Added |
| **Notebooks** | Jupyter | 7.4.2 | Added |

---

## 4. Full Project File Structure

```
MLOPS-PROJECT/
│
├── Dataset/
│   ├── Raw-Dataset/
│   │   └── neo.csv                     ← NASA NEO dataset (9 MB, 90,836 rows, 10 cols)
│   └── Processed-Dataset/              ← Populated after running Pre-processing.ipynb
│       ├── X_train_clf.csv             ← SMOTE-augmented scaled classification features
│       ├── X_val_clf.csv
│       ├── X_test_clf.csv              ← SEALED — only touched in Model-training.ipynb
│       ├── y_train_clf.csv
│       ├── y_val_clf.csv
│       ├── y_test_clf.csv
│       ├── X_train_reg.csv
│       ├── X_val_reg.csv
│       ├── X_test_reg.csv
│       ├── y_train_reg.csv
│       ├── y_val_reg.csv
│       ├── y_test_reg.csv
│       ├── scaler_clf.joblib           ← Fitted StandardScaler for clf features
│       └── scaler_reg.joblib           ← Fitted StandardScaler for reg features
│
├── model/
│   ├── Exploration.ipynb               ← PARTIALLY COMPLETE (EDA cells with output)
│   ├── Pre-processing.ipynb            ← FULLY WRITTEN (ready to run)
│   ├── Experimentation.ipynb           ← FULLY WRITTEN (ready to run)
│   ├── Model-training.ipynb            ← FULLY WRITTEN (ready to run)
│   ├── Final-model.ipynb               ← FULLY WRITTEN (ready to run)
│   ├── artifacts/                      ← Populated after running Model-training.ipynb
│   │   ├── classifier.joblib
│   │   ├── regressor.joblib
│   │   ├── scaler_clf.joblib
│   │   ├── scaler_reg.joblib
│   │   ├── feature_names.json
│   │   └── model_metadata.json
│   └── notebook-context/               ← Full context docs for each notebook
│       ├── Exploration-notebook.md
│       ├── Pre-processing-notebook.md
│       ├── Experimentaion-notebook.md  ← Note: filename has typo (Experimentaion)
│       ├── Model-training-notebook.md
│       └── Final-model-notebook.md
│
├── server/
│   ├── app.py                          ← FastAPI app with /api/predict endpoint
│   ├── readme.md
│   └── server-api.test/
│       └── api.test.py                 ← Empty stub
│
├── client/
│   ├── src/
│   │   ├── App.tsx                     ← Default Vite+React boilerplate (counter)
│   │   ├── main.tsx
│   │   ├── App.css
│   │   └── index.css
│   ├── package.json                    ← React 19.2, Vite 7.3, TypeScript 5.9
│   └── vite.config.ts
│
├── workers-sb/
│   ├── src/
│   │   ├── index.ts                    ← Express app with GET /health endpoint
│   │   └── worker-sb.test/
│   │       └── sb.test.ts              ← Empty stub
│   ├── package.json                    ← Express 5.2, TypeScript 5.9, dotenv
│   └── .env
│
├── sockets/
│   ├── src/
│   │   └── main.rs                     ← Placeholder: println!("Web socket layer will be added here")
│   └── Cargo.toml                      ← No dependencies yet
│
├── test-ml-model/
│   └── test.py                         ← Empty stub ("ML-MODEL test file")
│
├── .github/
│   └── workflows/
│       ├── main.yml                    ← CI/CD pipeline (ALL JOBS COMMENTED OUT)
│       └── project-context/            ← Empty placeholder .md files
│
├── docker-compose.yml                  ← 5 services: nginx, client, server, postgres, redis
├── Dockerfile.server                   ← Python 3.11-slim, installs requirements.txt
├── Dockerfile.client                   ← Node 22-alpine, npm install, vite dev server
├── nginx.conf                          ← Routes / → client:5173, /api → server:8000
├── requirements.txt                    ← 57 packages (FastAPI stack + ML stack)
├── Readme.md                           ← Project overview and setup instructions
├── project-flow.md                     ← Architecture diagram and component descriptions
└── what-i-have-done.md                 ← This file
```

---

## 5. Dataset

**File:** `Dataset/Raw-Dataset/neo.csv`
**Source:** NASA Near-Earth Object (NEO) dataset
**Size:** 9 MB, 90,836 rows, zero missing values

| Column | Type | Role |
|---|---|---|
| `id` | int64 | Dropped — database identifier, no signal |
| `name` | str | Dropped — free-text label, no signal |
| `est_diameter_min` | float64 | Engineered into `diameter_avg`, `diameter_ratio`, log transforms |
| `est_diameter_max` | float64 | Same as above |
| `relative_velocity` | float64 | Log-transformed → `log_relative_velocity` |
| `miss_distance` | float64 | **Regression target** (log-transformed during training) |
| `orbiting_body` | str | Dropped — always "Earth", zero variance |
| `sentry_object` | bool | Dropped — target leakage (NASA hazard flag) |
| `absolute_magnitude` | float64 | Kept as-is (well-behaved distribution) |
| `hazardous` | bool | **Classification target** (cast to int 0/1) |

**Class imbalance:** ~16–20% of records are hazardous — handled by SMOTE on training split.

**Key data characteristics observed during EDA:**
- All float features are right-skewed (max values orders of magnitude above median) → log1p transforms applied
- `orbiting_body` is 100% "Earth" → dropped
- `sentry_object` correlates with `hazardous` → dropped to prevent target leakage
- No missing values anywhere → no imputation needed

---

## 6. What Was Built Before This Session (Git History)

These components were committed to the repo before the current ML work session began:

| Commit | What Was Added |
|---|---|
| `f8da8a9` | Initial project setup |
| `5c3b9e1` | Readme.md |
| `1ff8f33` | requirements.txt (FastAPI stack only — no ML libs) |
| `b5bd87a` | Dataset folder created |
| `0a1bbb2` | `neo.csv` dataset added |
| `d4ae03d` | React client, FastAPI server scaffold, Docker files, nginx config |
| `90541b9` | `workers-sb` Express.js service added |
| `de7d6e5` | GitHub Actions CI/CD workflow added (commented out) |
| `2dbcc45` | Empty test file stubs added (test.py, api.test.py, sb.test.ts) |
| `5c6a998` | Rust WebSocket placeholder added |
| `fcb9937` | Project context placeholder files added |

**State at the start of this session:**
- FastAPI had 2 trivial GET endpoints — no ML
- `Exploration.ipynb` had 9 cells with EDA output (head, tail, info, describe, isnull)
- The other 4 notebooks were completely empty (0 bytes)
- No ML dependencies in `requirements.txt`
- No `model/artifacts/` directory
- No `notebook-context/` markdown files (just placeholder stubs)

---

## 7. What Was Done in This Session (Claude)

### 7.1 ML Pipeline Planning
A full supervised ML plan was designed and documented covering:
- Task definition: binary classification (`hazardous`) + regression (`miss_distance`)
- Feature engineering strategy (log-transforms, derived features, drop rationale)
- Split strategy (stratified 70/15/15)
- Class imbalance handling (SMOTE on train only)
- Baseline model candidates for both tasks
- Hyperparameter tuning approach (Optuna/TPE, 50 trials)
- Model serialisation and FastAPI integration pattern

### 7.2 `requirements.txt` — 8 New ML Dependencies Added
```
scikit-learn==1.6.1
matplotlib==3.10.1
seaborn==0.13.2
joblib==1.4.2
optuna==4.2.1
imbalanced-learn==0.13.0
notebook==7.4.2
ipykernel==6.29.5
```

### 7.3 `model/Pre-processing.ipynb` — Fully Implemented
All 19 cells implemented across 9 sections:
1. Imports and configuration (paths, RANDOM_STATE=42)
2. Load + assert-validate raw CSV
3. Drop 4 columns (id, name, orbiting_body, sentry_object)
4. Feature engineering: `diameter_avg`, `diameter_ratio`, 4 log1p transforms; defensive assertions for Inf/NaN
5. Encode `hazardous` bool → int
6. Define `CLF_FEATURES` (7) and `REG_FEATURES` (6) — regression excludes its own target
7. Stratified 70/15/15 split (two `train_test_split` calls with `stratify=y_clf`)
8. SMOTE oversampling on training split only
9. `StandardScaler` fit on train, transform val/test — two separate scalers
10. Save 12 CSV splits + 2 `.joblib` scalers to `Dataset/Processed-Dataset/`

### 7.4 `model/Experimentation.ipynb` — Fully Implemented
All 17 cells implemented:
- 4 classification baselines: `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`
- 3 regression baselines: `LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`
- Results tables sorted by F1 (clf) and R² (reg)
- 5-fold `StratifiedKFold` cross-validation on top 2 classifiers
- 5-fold `KFold` cross-validation on top 2 regressors
- 4 visualisation cells: confusion matrices, ROC + PR curves, feature importance, residual plots
- Model selection decision markdown cell

### 7.5 `model/Model-training.ipynb` — Fully Implemented
All 14 cells implemented:
- Loads all 12 CSVs, builds train+val combined sets
- Optuna `clf_objective`: 50 trials, maximises F1-weighted on val set
- Optuna `reg_objective`: 50 trials, minimises RMSE on val set (log-space)
- Search space for both: `n_estimators` [100–500], `max_depth` [3–8], `learning_rate` [1e-3, 0.3 log], `min_samples_split` [2–20], `subsample` [0.6–1.0]
- Final training on train+val combined (85% of data)
- One-time test set evaluation: classification report + AUC; R², RMSE in log and km
- Saves all 6 artifacts to `model/artifacts/`

### 7.6 `model/Final-model.ipynb` — Fully Implemented
All 12 cells implemented:
- Loads all 6 artifacts and verifies them
- `predict_asteroid()` function: raw inputs → feature engineering → scale → predict → inverse transform → returns `{hazardous, hazardous_probability, miss_distance_km}`
- Smoke test with a concrete example
- 3-panel final plot: confusion matrix (raw counts), ROC curve with AUC, predicted vs. actual scatter (km scale)
- Feature importance comparison chart (classifier vs. regressor side-by-side)
- FastAPI reference implementation as a documentation cell
- Final metrics summary table read from `model_metadata.json`

### 7.7 `server/app.py` — Updated with Prediction Endpoint
Replaced the trivial 14-line scaffold with a full FastAPI application:
- `AsteroidInput` Pydantic model (5 input fields with float types)
- `PredictionOutput` Pydantic model (3 output fields)
- `lifespan` async context manager: loads 5 artifacts at startup, graceful warning if artifacts missing
- `GET /` — health check (unchanged)
- `GET /items/{item_id}` — unchanged
- `POST /api/predict` — full inference pipeline:
  - Feature engineering (mirrors Pre-processing.ipynb exactly)
  - DataFrame construction with named columns in correct order
  - Scale with loaded scalers
  - Predict classification and regression
  - Inverse log-transform regression output
  - Returns `PredictionOutput` with rounded values

### 7.8 `docker-compose.yml` — Volume Mount Added
Added `./model/artifacts:/app/model/artifacts` to the server service's volumes so the trained model files are accessible inside the Docker container without rebuilding the image.

### 7.9 `model/notebook-context/` — All 5 Files Fully Written
Replaced empty placeholder stubs with comprehensive documentation:

| File | Lines | Contents |
|---|---|---|
| `Exploration-notebook.md` | 156 | Cell-by-cell EDA explanation, theory of EDA, findings table |
| `Pre-processing-notebook.md` | 303 | All 19 cells explained, log-transform/SMOTE/scaling theory, data flow diagram |
| `Experimentaion-notebook.md` | 233 | All 17 cells, theory for each algorithm, metric definitions, visualisation theory |
| `Model-training-notebook.md` | 269 | Optuna/Bayesian optimisation theory, hyperparameter table, sealed test set rationale |
| `Final-model-notebook.md` | 270 | predict_asteroid explained step-by-step, inverse transform theory, API contract table |

---

## 8. Component Status

| Component | File(s) | Status | Notes |
|---|---|---|---|
| **Dataset** | `Dataset/Raw-Dataset/neo.csv` | Ready | 90,836 rows, zero nulls |
| **EDA Notebook** | `model/Exploration.ipynb` | Partial | 9 cells with output, no changes needed |
| **Pre-processing Notebook** | `model/Pre-processing.ipynb` | Written, not run | Run this first to generate processed splits |
| **Experimentation Notebook** | `model/Experimentation.ipynb` | Written, not run | Depends on Pre-processing output |
| **Model Training Notebook** | `model/Model-training.ipynb` | Written, not run | Depends on Experimentation decision |
| **Final Model Notebook** | `model/Final-model.ipynb` | Written, not run | Depends on model/artifacts/ |
| **Notebook Context Docs** | `model/notebook-context/*.md` | Complete | All 5 files fully written |
| **FastAPI Server** | `server/app.py` | Complete | Has /api/predict, loads models at startup |
| **Python Dependencies** | `requirements.txt` | Complete | 57 packages including full ML stack |
| **Docker Compose** | `docker-compose.yml` | Complete | 5 services + model artifact volume mount |
| **Nginx Config** | `nginx.conf` | Complete | Routes / and /api correctly |
| **Dockerfiles** | `Dockerfile.server`, `Dockerfile.client` | Complete | Python 3.11-slim, Node 22-alpine |
| **React Frontend** | `client/src/App.tsx` | Scaffold | Default Vite boilerplate, no ML UI yet |
| **Worker Service** | `workers-sb/src/index.ts` | Scaffold | Only GET /health endpoint |
| **Rust WebSocket** | `sockets/src/main.rs` | Placeholder | Prints a string, nothing implemented |
| **CI/CD Pipeline** | `.github/workflows/main.yml` | Configured but disabled | All jobs commented out |
| **PostgreSQL** | docker-compose.yml | In Docker | Not connected to app code |
| **Redis** | docker-compose.yml | In Docker | Not connected to app code |
| **ML Model Tests** | `test-ml-model/test.py` | Empty stub | Needs to be implemented |
| **API Tests** | `server/server-api.test/api.test.py` | Empty stub | Needs to be implemented |
| **model/artifacts/** | (directory) | Not yet created | Created when Model-training.ipynb is run |
| **Dataset/Processed-Dataset/** | (CSVs) | Not yet populated | Created when Pre-processing.ipynb is run |

---

## 9. ML Pipeline — Detailed Walkthrough

### Feature Engineering (applied in Pre-processing, replicated in server/app.py)

```python
# Input: raw asteroid observation
est_diameter_min, est_diameter_max, relative_velocity, absolute_magnitude, miss_distance

# Derived features
diameter_avg          = (est_diameter_min + est_diameter_max) / 2
diameter_ratio        = est_diameter_max / est_diameter_min
log_diameter_avg      = np.log1p(diameter_avg)
log_diameter_ratio    = np.log1p(diameter_ratio)
log_relative_velocity = np.log1p(relative_velocity)
log_miss_distance     = np.log1p(miss_distance)  # regression target / clf feature
```

### Feature Sets

**Classification features (7):**
`log_diameter_avg`, `log_diameter_ratio`, `log_relative_velocity`, `log_miss_distance`, `absolute_magnitude`, `diameter_avg`, `diameter_ratio`

**Regression features (6):**
Same minus `log_miss_distance` (it is the regression target, so it cannot be a feature)

### Split Strategy
- Stratified 70/15/15 using two `train_test_split` calls
- Stratification on `hazardous` label throughout
- Regression uses same row indices as classification for consistency

### Class Imbalance Handling
- SMOTE applied to classification training split only
- Creates synthetic minority-class (hazardous=1) samples by interpolating in feature space
- Val/test sets are never oversampled — they reflect the real class distribution

### Scaling
- `StandardScaler` fit on training data only
- Applied to val/test using same fitted parameters
- Regression target (`log_miss_distance`) is NOT scaled — already near-normal after log transform
- Scalers saved as `scaler_clf.joblib` and `scaler_reg.joblib`

### Models
Both tasks use `GradientBoostingClassifier` / `GradientBoostingRegressor` from scikit-learn:
- Sequential ensemble of decision trees
- Each tree corrects residual errors of previous ensemble
- Configured with Optuna-tuned hyperparameters

### Tuning
- Optuna TPE sampler, 50 trials per model
- Classification objective: maximise F1-weighted on val set
- Regression objective: minimise RMSE on val set (log-space)
- Final model trained on train+val combined (85%)

### Artifacts Produced
```
model/artifacts/
├── classifier.joblib      ← GradientBoostingClassifier (tuned, final)
├── regressor.joblib       ← GradientBoostingRegressor (tuned, final)
├── scaler_clf.joblib      ← Must be used at inference for clf features
├── scaler_reg.joblib      ← Must be used at inference for reg features
├── feature_names.json     ← Ordered feature lists for both tasks
└── model_metadata.json    ← Training timestamp, hyperparams, test metrics
```

---

## 10. API Contract

### `POST /api/predict`

**Request body:**
```json
{
  "est_diameter_min":   0.12,
  "est_diameter_max":   0.27,
  "relative_velocity":  48000.0,
  "absolute_magnitude": 22.1,
  "miss_distance":      14500000.0
}
```

**Response:**
```json
{
  "hazardous":             false,
  "hazardous_probability": 0.23,
  "miss_distance_km":      42300000.0
}
```

**Field meanings:**
- `hazardous` — binary classifier prediction (is this a Potentially Hazardous Object?)
- `hazardous_probability` — classifier confidence between 0 and 1 (usable for risk gauge UI)
- `miss_distance_km` — regressor prediction of closest approach distance in km (inverse log-transform applied)

**Error behaviour:**
- Returns `503 Service Unavailable` with `"Models not loaded. Run training notebooks first."` if `model/artifacts/` is missing

**Other endpoints:**
- `GET /` → `{"status": "success", "message": "MLOps Server is Live!"}`
- `GET /items/{item_id}` → `{"item_id": ..., "category": "ML-Model"}`
- `GET /docs` → Swagger UI (proxied by Nginx)

---

## 11. Infrastructure and Deployment

### Docker Compose Services

| Service | Image | Port | Notes |
|---|---|---|---|
| `nginx` | nginx:alpine | 80 (host) | Gateway, routes to client and server |
| `client` | Dockerfile.client | 5173 (internal) | Node 22-alpine, Vite dev server |
| `server` | Dockerfile.server | 8000 (internal) | Python 3.11-slim, uvicorn |
| `postgres` | postgres:16-alpine | 5432 (internal) | Password: password123 |
| `redis` | redis:7-alpine | 6379 (internal) | Default config |

### Volume Mounts (server service)
```yaml
- ./Dataset:/app/Dataset              # Raw and processed datasets
- ./model/artifacts:/app/model/artifacts  # Trained model files
```

### Nginx Routing
```
GET  /           → http://client:5173     (React app, with WebSocket upgrade for HMR)
GET  /api*       → http://server:8000     (FastAPI)
GET  /docs       → http://server:8000/docs
GET  /openapi.json → http://server:8000/openapi.json
```

### CI/CD Pipeline (GitHub Actions — `main.yml`)
Three jobs are defined but all are commented out:
1. **test-server** — Python 3.11 setup, `pip install -r requirements.txt`, `flake8 server/`
2. **test-client** — Node 22 setup, `npm install`, `npm run build`
3. **build-and-push** — builds Docker images and pushes to DockerHub (requires `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets)

---

## 12. How to Run the Project

### Prerequisites
- Python 3.11+
- Node.js 22+
- Rust (for sockets, optional)
- Docker + Docker Compose

### Step 1 — Install Python Dependencies
```bash
cd /home/enjin/projects/MLOPS-PROJECT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Run ML Notebooks in Order

Open Jupyter and run each notebook completely from top to bottom:

```bash
cd model/
jupyter notebook
```

**Run in this order:**
1. `Exploration.ipynb` — already has output, review only
2. `Pre-processing.ipynb` — generates `Dataset/Processed-Dataset/` with 14 files
3. `Experimentation.ipynb` — trains baselines, compare metrics, review model selection cell
4. `Model-training.ipynb` — runs 100 Optuna trials (50 per model), saves `model/artifacts/`
5. `Final-model.ipynb` — loads artifacts, smoke test, final plots

After Step 2, `model/artifacts/` will contain all 6 deployment files.

### Step 3 — Run FastAPI Server Locally (without Docker)
```bash
cd server/
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Test the prediction endpoint:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "est_diameter_min": 0.12,
    "est_diameter_max": 0.27,
    "relative_velocity": 48000,
    "absolute_magnitude": 22.1,
    "miss_distance": 14500000
  }'
```

### Step 4 — Run with Docker Compose
```bash
docker compose up --build
```
- React app: http://localhost/
- API: http://localhost/api/predict
- Swagger docs: http://localhost/docs

**Note:** The Docker server container looks for artifacts at `/app/model/artifacts/` (mounted from `./model/artifacts/`). Run the notebooks first (Step 2) before starting Docker.

---

## 13. What Still Needs to Be Done

### High Priority (core functionality)
- [ ] **Run the notebooks** — `Pre-processing.ipynb` through `Final-model.ipynb` have never been executed; all code is written but no models are trained yet
- [x] **Connect PostgreSQL to the server** — `asyncpg` pool added; predictions table auto-created on startup; each prediction logged
- [x] **Connect Redis to the worker or server** — Redis caching added to both the server (prediction cache, 1-hour TTL) and the worker (cache stats/flush endpoints)
- [x] **Fix Dockerfile.server CMD** — fixed to `uvicorn app:app`

### Medium Priority (features)
- [x] **Implement React UI** — `client/src/App.tsx` replaced with an asteroid prediction form with dark-mode UI and hazard/safe result display
- [x] **Implement the Worker service** — `workers-sb/src/index.ts` now connects to Redis and exposes `GET /cache/stats` and `DELETE /cache/flush`
- [x] **Implement Rust WebSocket layer** — `sockets/src/main.rs` implements a tokio-tungstenite broadcast WebSocket server on port 9001
- [x] **Write ML model tests** — `test-ml-model/test.py` covers feature engineering correctness, artifact loading, classifier/regressor output shapes
- [x] **Write API tests** — `server/server-api.test/api.test.py` covers health endpoints, input validation (422s), inference shape, determinism, and 503 path

### Low Priority (ops)
- [x] **Enable CI/CD pipeline** — `main.yml` uncommented and expanded with 4 jobs: test-server, test-client, test-worker, build-and-push
- [x] **Add prediction logging to PostgreSQL** — done in `server/app.py` lifespan + predict endpoint
- [x] **Add Redis caching** — done in `server/app.py` with SHA-256 keyed cache (1-hour TTL)
- [x] **Fix typo in context filename** — renamed to `Experimentation-notebook.md`
- [x] **Configure Sentry error tracking** — `sentry_sdk.init()` called at startup when `SENTRY_DSN` env var is set

---

## 14. Key Design Decisions and Rationale

| Decision | Rationale |
|---|---|
| Drop `sentry_object` | Target leakage — NASA's Sentry flag is based on the same hazard assessment as `hazardous`. Including it would inflate all metrics while giving no real predictive power at inference. |
| Drop `orbiting_body` | All 90,836 rows have `orbiting_body = "Earth"`. Zero variance adds no information and only noise. |
| Log1p transform on skewed features | Diameter ranges from 0.0006 km to 84 km — 5 orders of magnitude. Linear models and distance-based models perform poorly on such ranges. Log1p compresses the scale while preserving ordering and handling zero values safely. |
| Regression target: `miss_distance` in log-space | Training the regressor on `log1p(miss_distance)` rather than raw km makes the target near-Gaussian and reduces the influence of extreme outliers. Output is inverse-transformed at prediction time. |
| SMOTE only on training split | Applying oversampling before the split leaks synthetic samples into the validation and test sets, making metrics optimistic. SMOTE must only see and augment training data. |
| StandardScaler fit on train only | The scaler's mean/std must be computed from training data only. Fitting on all data (including val/test) would leak test statistics into the preprocessing, violating the train/test separation. |
| Separate scalers for clf and reg | The classification task has 7 features; the regression task has 6 (excludes `log_miss_distance`). Using a single scaler would require careful column management — two separate scalers is cleaner and less error-prone. |
| Optuna over GridSearchCV | Bayesian optimisation (TPE) finds good hyperparameters in 50 trials. Grid search over the same 5-dimensional space with even 3 values per dimension would require 3⁵ = 243 evaluations. On 90K rows, each trial is slow. |
| Final training on train+val combined | After tuning, retraining on 85% of data (train+val) gives the model more examples to learn from. The test set stays sealed so the final evaluation is unbiased. |
| Models loaded at FastAPI startup (lifespan) | Loading joblib models takes 100ms–2s. Loading per-request would make the API unacceptably slow. The `lifespan` context manager loads once at startup and keeps models in-process memory for the server's lifetime. |
| Feature names saved as JSON | At inference time, the server constructs a pandas DataFrame. Column order must exactly match training order or predictions are silently wrong. Saving the ordered feature lists removes this risk. |
| `model/artifacts/` as a Docker volume mount | Separates trained model artifacts from application code. Models can be updated (retrained) without rebuilding the Docker image — just replace the files in the mounted directory and restart the container. |
