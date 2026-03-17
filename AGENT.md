# AGENT.md — MedSegAPI Project Instructions

## Project Overview
MedSegAPI is a production medical image segmentation platform serving hybrid CNN-ViT
lung segmentation models via FastAPI. The project demonstrates end-to-end ML engineering:
research model → API → Docker → CI/CD → monitoring → continuous training.

## Tech Stack
- **Language:** Python 3.11
- **Package Manager:** uv (NOT pip, NOT conda for project deps)
- **Linter/Formatter:** ruff (NOT black, NOT flake8)
- **API:** FastAPI + uvicorn
- **ML:** PyTorch (use CPU-only for API serving; GPU via conda ml_env for training)
- **Model Registry:** MLflow
- **Data Versioning:** DVC
- **Database:** SQLite (production + dev), PostgreSQL available in compose stack
- **Testing:** pytest + pytest-asyncio + httpx (async API tests)
- **Container:** Docker with multi-stage builds (ARM64 + x86_64)
- **CI/CD:** GitHub Actions
- **Deployment:** Oracle Cloud A1.Flex (ARM64 Ampere, production)
- **Reverse Proxy:** Caddy with automatic HTTPS (Let's Encrypt)
- **Domain:** lungmedseg.duckdns.org (DuckDNS free DNS)
- **Monitoring:** Streamlit dashboard + Prometheus metrics

## Environment
- WSL2 Ubuntu 22.04 on Windows 11
- WSL username: home_laboratory
- WSL hostname: kppc
- Home directory: /home/home_laboratory
- Conda environment `ml_env` available for GPU training (PyTorch nightly cu128)
- uv for project dependency management (API serving, all non-GPU work)
- GPU: NVIDIA GeForce RTX 5070 Ti 16GB GDDR7
- Thesis project location: ~/AIT_LungSegmentation/

## Coding Standards
1. Always use `uv add` for new dependencies, never pip install
2. Run `ruff check . && ruff format .` before every commit
3. Every new module must have a corresponding test in tests/
4. Use Python logging module, NEVER print statements
5. Type hints on ALL function signatures
6. Docstrings on all public functions (Google style)
7. Config via environment variables + YAML files, never hardcoded paths
8. All API responses use Pydantic models
9. Git commits: conventional commits (feat:, fix:, docs:, test:, ci:, refactor:)
10. Line length: 100 characters max
11. Import sorting: isort-compatible via ruff

## Model Architecture Reference
Three model variants from the thesis:

### CNN-only
- Architecture: ResNet-18 encoder + U-Net decoder
- Parameters: ~15-18M
- Input: 512×512 chest X-ray (3-channel RGB)
- Output: binary segmentation mask

### ViT-only
- Architecture: DeiT-Tiny + progressive upsampling decoder
- Parameters: ~8-10M
- Input: 512×512 chest X-ray (3-channel RGB)
- Output: binary segmentation mask

### Hybrid (primary model)
- Architecture: CNN encoder + ViT encoder + cross-attention fusion + decoder
- Parameters: ~4.2M (optimized)
- Input: 512×512 chest X-ray (3-channel RGB)
- Output: binary segmentation mask
- Best Dice: 96.65% on Montgomery dataset
- Cross-dataset: JSRT 95.18%, Shenzhen 94.82%

### Training Config (thesis defaults)
- Learning rate: 0.0005
- Batch size: 32
- Image size: 512×512
- Loss: Focal (α=0.25, γ=2.0) + Dice + Boundary (weight=0.3)
- Optimizer: Adam
- Scheduler: CosineAnnealingLR
- Early stopping patience: 10

## Key Datasets
- Montgomery (MC): 138 CXR images (80 normal, 58 TB) — primary test set
- Shenzhen: 662 CXR images (326 normal, 336 TB)
- JSRT: 247 CXR images
- Qatar COVID-QU-Ex: 33,920 images (for extended training)

## API Design
- POST /api/v1/predict — Single image segmentation
- POST /api/v1/batch — Batch processing (up to 10 images)
- GET  /api/v1/models — List available models with metadata
- GET  /api/v1/models/{model_id}/info — Model details + metrics
- GET  /api/v1/health — Liveness check
- GET  /api/v1/ready — Readiness check (model loaded)
- GET  /api/v1/metrics — Prometheus-compatible metrics
- GET  /api/v1/ab-results — A/B test comparison results

## Server Access
```bash
# SSH to Oracle Cloud production server
ssh -i ~/.ssh/ssh-key-2026-03-17.key ubuntu@140.245.113.52
cd ~/medseg-api

# Deploy updates
git pull && sudo docker compose -f docker-compose.prod.yml up -d --build api

# Manage API keys (always restart after)
python3 scripts/create_api_key.py --name "guest" --max-uses 3
sudo docker compose -f docker-compose.prod.yml restart api

# View logs
sudo docker compose -f docker-compose.prod.yml logs --tail 50 api
```

## Project Structure
```
medseg-api/
├── AGENT.md                       # Project instructions & standards
├── CLAUDE.md                      # Extended context for AI assistants
├── README.md                      # Main documentation
├── CHANGELOG.md                   # Version history
├── pyproject.toml                 # Dependencies (uv)
├── uv.lock
├── Dockerfile                     # Multi-stage build (ARM64 + x86_64)
├── docker-compose.yml             # Dev stack
├── docker-compose.prod.yml        # Prod stack (Caddy + API + PostgreSQL)
├── Caddyfile                      # Reverse proxy (auto HTTPS)
├── .github/workflows/
│   ├── ci.yml
│   ├── cd.yml
│   └── retrain.yml
├── .env.example
├── .gitignore
├── .dockerignore
├── .ruff.toml
├── configs/
│   ├── model_registry.yaml        # Model versions & metrics
│   ├── serving.yaml               # Server & model config
│   └── api_keys.json              # SHA-256 hashed API keys (.gitignored)
├── frontend/
│   └── index.html                 # Interactive web viewer (canvas-based)
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app factory & lifespan
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── predict.py         # POST /predict, /batch
│   │   │   ├── models.py          # GET /models
│   │   │   ├── health.py          # GET /health, /ready
│   │   │   └── monitoring.py      # GET /metrics, /predictions
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py            # API key auth (master + guest keys)
│   │   │   ├── rate_limit.py      # Per-key rate limiting
│   │   │   └── logging_mw.py      # Request logging
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── request.py
│   │       └── response.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── architectures/
│   │   │   ├── __init__.py
│   │   │   ├── cnn_model.py       # ResNet-18 + U-Net
│   │   │   ├── vit_model.py       # DeiT-Tiny + Decoder
│   │   │   └── hybrid_model.py    # CNN-ViT + Cross-Attention
│   │   ├── registry.py            # Model versioning & hot-swap
│   │   ├── inference.py           # Preprocessing & prediction
│   │   └── explainability.py      # Grad-CAM / SHAP
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── dataset.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── augmentations.py
│   │   ├── callbacks.py
│   │   └── evaluation.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── prediction_logger.py   # SQLite prediction logging
│   │   ├── logger.py
│   │   ├── drift.py
│   │   └── dashboard.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── dicom.py
│       └── image.py
├── scripts/
│   ├── create_api_key.py          # Generate/list/revoke API keys
│   ├── setup_models.py            # Copy thesis weights + validate
│   ├── train.py                   # Training entrypoint
│   ├── retrain.py                 # Continuous retraining
│   ├── evaluate.py                # Model evaluation
│   ├── register_model.py          # Register in MLflow
│   ├── export_onnx.py             # Export to ONNX
│   ├── download_data.py           # Dataset download
│   ├── preprocess_data.py         # Data preprocessing
│   ├── add_new_data.py            # Add new training data
│   ├── predict_and_save.py        # Batch prediction
│   └── deploy.sh                  # Deployment automation
├── monitoring/
│   └── streamlit_app.py           # Monitoring dashboard
├── tests/                         # 152 tests (pytest)
│   ├── conftest.py
│   ├── test_api/
│   ├── test_models/
│   ├── test_monitoring/
│   ├── test_training/
│   └── test_utils/
├── models/                        # Model weights (.gitignored)
├── data/                          # Training data (.gitignored)
└── results/                       # Evaluation outputs
```

## When Making Changes
1. Read relevant existing code first before editing
2. Write tests before or alongside implementation
3. Run the full test suite: `uv run pytest -v`
4. Run linter: `uv run ruff check . && uv run ruff format .`
5. If touching API: verify with `uv run uvicorn src.api.main:app --reload`
6. If touching models: verify with a quick inference test on a dummy tensor
7. Commit with conventional commit message
8. Never commit .pth files, data files, or .env to git
