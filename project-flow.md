# MLOPS-PROJECT — Comprehensive Project Overview

## Architecture

```
Internet/User
    │
    └─→ Nginx (Port 80)
         │
         ├─→ React Client (Vite Dev Server, Port 5173)
         │   ├─ Makes API calls to backend
         │   └─ Intended WebSocket connections
         │
         └─→ FastAPI Server (Port 8000)
             ├─ Reads datasets from /Dataset mount
             ├─ Validates requests with Pydantic
             ├─ Queries PostgreSQL database
             ├─ Caches results in Redis
             ├─ May delegate to Worker service
             └─ Returns predictions/responses

        ┌──────────────────────────┐
        │   Express Worker (Port 3000)
        │   - Background processing
        │   - Redis caching
        │   - Secondary business logic
        └──────────────────────────┘
               ↑
               │ (Async communication)
               │
        ┌──────────────────────────┐
        │   Rust WebSocket Layer
        │   - Real-time updates
        │   - Bidirectional streaming
        │   (Currently placeholder)
        └──────────────────────────┘

┌─────────────────────────────────────────┐
│   ML Model Pipeline (Development)       │
│   5 Jupyter Notebooks:                  │
│   - Exploration (partially complete)    │
│   - Pre-processing (empty)              │
│   - Experimentation (empty)             │
│   - Model Training (empty)              │
│   - Final Model (empty)                 │
│                                         │
│   Uses: pandas, numpy, scikit-learn     │
│   Data: /Dataset/Raw & Processed        │
└─────────────────────────────────────────┘
```

---

## Directory Structure

```
MLOPS-PROJECT/
├── .github/workflows/
│   └── main.yml                    # CI/CD pipeline configuration
├── client/                         # React + TypeScript frontend
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── index.css
│   │   ├── App.css
│   │   └── assets/
│   ├── public/
│   ├── index.html
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── eslint.config.js
│   ├── package.json
│   └── README.md
├── server/                         # FastAPI backend
│   ├── app.py                      # Main FastAPI application
│   ├── readme.md
│   └── server-api.test/
│       └── api.test.py
├── workers-sb/                     # Express.js worker service
│   ├── src/
│   │   ├── index.ts
│   │   └── worker-sb.test/
│   ├── dist/
│   ├── package.json
│   ├── tsconfig.json
│   └── .env                        # PORT=3000
├── sockets/                        # Rust WebSocket layer
│   ├── src/
│   │   └── main.rs
│   ├── Cargo.toml
│   └── Cargo.lock
├── model/                          # Jupyter notebooks for ML pipeline
│   ├── Exploration.ipynb           # Data exploration (partially complete)
│   ├── Pre-processing.ipynb        # Data preprocessing (empty)
│   ├── Experimentation.ipynb       # Model experimentation (empty)
│   ├── Model-training.ipynb        # Model training (empty)
│   └── Final-model.ipynb           # Results & evaluation (empty)
├── Dataset/                        # ML datasets
│   ├── Raw-Dataset/
│   │   ├── dataset.csv
│   │   └── neo.csv
│   └── Processed-Dataset/
│       └── dataset.csv
├── test-ml-model/
│   └── test.py                     # ML model tests
├── Dockerfile.client               # Node 22 Alpine container
├── Dockerfile.server               # Python 3.11 Slim container
├── docker-compose.yml              # Service orchestration
├── nginx.conf                      # Reverse proxy configuration
├── requirements.txt                # Python dependencies
├── Readme.md                       # Main project documentation
└── .dockerignore
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19.2, TypeScript 5.9, Vite 7.3 |
| Primary Backend | FastAPI 0.135, Python 3.11, Pydantic 2.12, Uvicorn 0.41 |
| Worker Backend | Express.js 5.2, TypeScript 5.9 |
| WebSocket | Rust (placeholder) |
| Database | PostgreSQL 16 Alpine |
| Cache | Redis 7 Alpine |
| Gateway | Nginx |
| ML / Data | pandas 3.0, numpy 2.4, Jupyter notebooks |
| Error Tracking | Sentry SDK 2.54 |
| Infra | Docker Compose, GitHub Actions |

---

## Major Components

### Frontend (`client/`)
- React 19.2 with TypeScript for the user interface
- Vite as the build tool and dev server (port 5173)
- ESLint for code quality
- Planned: axios for API calls, Zod for validation, TanStack/Recoil for state, Shadcn for components, Three.js for 3D visualizations

### Primary Backend (`server/`)
- FastAPI REST API exposing ML model predictions
- Pydantic for input validation
- Uvicorn ASGI server on port 8000
- Planned: PostgreSQL for relational data, Pinecone for model importing

### Worker Service (`workers-sb/`)
- Express.js (TypeScript) secondary processing service on port 3000
- Handles background and async tasks
- Planned: Redis caching integration

### WebSocket Layer (`sockets/`)
- Rust-based implementation (currently a placeholder)
- Intended for real-time bidirectional communication and live updates

### ML Model Pipeline (`model/`)
Five-phase Jupyter notebook workflow:
1. **Exploration** — Initial data analysis and visualization *(partially complete)*
2. **Pre-processing** — Data cleaning, transformation, and splitting *(empty)*
3. **Experimentation** — Model comparison and hyperparameter tuning *(empty)*
4. **Model Training** — Final model optimization *(empty)*
5. **Final Model** — Evaluation, reporting, and deployment prep *(empty)*

### Datasets (`Dataset/`)
- Raw datasets: `dataset.csv`, `neo.csv`
- Processed/cleaned datasets ready for training

---

## Key Configuration Files

### `docker-compose.yml`
Defines 5 services: `nginx`, `client`, `server`, `postgres`, `redis`
- Nginx gateway on port 80 routing traffic to client/server
- Client: Vite dev server (port 5173 internally)
- Server: FastAPI (port 8000 internally)
- PostgreSQL with persistent volume (`postgres_data`)
- Redis for caching
- Server has mounted Dataset volume for access to training data

### `nginx.conf`
Reverse proxy on port 80:
- `/` → client (Vite dev server on `:5173`)
- `/api` → server (FastAPI on `:8000`)
- `/docs` → server API documentation
- `/openapi.json` → OpenAPI schema

### `Dockerfile.client`
- Base: `node:22-alpine`
- Installs dependencies with npm
- Exposes port 5173
- Starts Vite with `--host 0.0.0.0`

### `Dockerfile.server`
- Base: `python:3.11-slim`
- Installs `build-essential` for ML library compilation
- Installs from `requirements.txt`
- Exposes port 8000
- Starts Uvicorn on `0.0.0.0:8000`

### `requirements.txt`
47 Python packages including:
- FastAPI stack: FastAPI, Starlette, Uvicorn, Pydantic
- Data science: numpy, pandas
- Async utilities: anyio, httpx, uvloop
- Error tracking: sentry-sdk
- Validation: email-validator, pydantic-extra-types

---

## CI/CD Pipeline (`.github/workflows/main.yml`)

**Currently commented out.** Configured for:

**Triggers:** Push and pull requests to `main` branch

**Jobs:**
1. **test-server** — Sets up Python 3.11, installs deps, runs flake8 linting on `server/`
2. **test-client** — Sets up Node.js 22, installs deps, builds TypeScript and Vite bundle
3. **build-and-push** — Runs only if both test jobs pass; builds and pushes Docker images to DockerHub

---

## Component Interaction Flow

```
1. Data Scientists work in Jupyter notebooks (model/) to develop the ML model
2. Backend developer exposes the trained model via FastAPI endpoints (server/)
3. Frontend developer builds UI in React (client/) that calls API endpoints
4. Worker service (workers-sb/) handles background tasks and caching
5. WebSocket layer (sockets/) enables real-time features (WIP)
6. Docker Compose orchestrates all services locally
7. GitHub Actions (when enabled) tests and builds Docker images on push
```

---

## Project Status (as of 2026-03-23)

| Component | Status |
|---|---|
| Infrastructure (Docker, Nginx) | Fully configured |
| CI/CD (GitHub Actions) | Configured but disabled (commented out) |
| ML Exploration Notebook | Partially complete |
| ML Pre-processing → Final Model | Empty / not started |
| Frontend | Basic React scaffold |
| FastAPI Backend | Basic endpoints, sample structure |
| Express Worker | Health check endpoint only |
| WebSocket (Rust) | Placeholder only |
| PostgreSQL integration | In Docker Compose, not wired to app code |
| Redis integration | In Docker Compose, not wired to app code |

The project has all infrastructure scaffolded. The main work ahead is completing the ML model training pipeline and integrating the trained model with the API layer.
