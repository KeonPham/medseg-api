# Changelog

All notable changes to MedSegAPI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-03-17

### Added

- **Cloud deployment** — Deployed to Oracle Cloud A1.Flex (ARM64 Ampere, 2 OCPU,
  12GB RAM) with Docker Compose production stack
- **Automatic HTTPS** — Caddy reverse proxy with Let's Encrypt auto-provisioned TLS
  certificates via DuckDNS free domain (lungmedseg.duckdns.org)
- **Guest API keys** — New key type with configurable usage limits (`--max-uses`),
  tracked via a per-key counter in `api_keys.json`, returning 403 when exhausted
- **API key management CLI** — Extended `scripts/create_api_key.py` with `--list`
  (tabular overview of all keys), `--revoke` (disable a key by name), and
  `--max-uses` (create guest keys with limited predictions)
- **Interactive frontend viewer** — Canvas-based viewer with zoom, pan, brightness,
  contrast, and layer toggle (original / mask / overlay) served at `/`
- **Production docker-compose** — `docker-compose.prod.yml` with Caddy, API (8GB
  memory limit), and PostgreSQL; ARM64-aware Dockerfile with CPU-only PyTorch

### Changed

- Dockerfile now copies `README.md` into builder stage (required by hatchling)
- Default database switched to SQLite (`sqlite:///./data/predictions.db`) for
  simpler single-node deployment
- Frontend removed prediction history panel for cleaner UI
- Frontend default view changed to overlay layer with auto-adjusted brightness/contrast

### Fixed

- Black screen on interactive viewer caused by `contrast=1` being divided by 100
  in render loop (now defaults to `contrast=120`)
- Canvas dimensions computed as 0x0 when viewer section was `display:none` during
  `fitToView()` — now section is shown before image loading
- Guest key validation returning HTTP 500 when `_persist()` failed due to file
  permissions — now wrapped in try/except with warning log

## [1.0.0] - 2026-03-15

### Added

- **API serving** — FastAPI application serving three lung segmentation model variants
  (CNN-only, ViT-only, Hybrid CNN-ViT) with single and batch prediction endpoints
- **Model architectures** — Ported thesis models: ResNet-18 + U-Net (CNN), DeiT-Tiny +
  progressive decoder (ViT), CNN-ViT with cross-attention fusion (Hybrid, 96.65% Dice)
- **Model registry** — YAML-based model versioning with hot-swap capability and
  MLflow integration for experiment tracking
- **Inference pipeline** — Image preprocessing, DICOM support, mask overlay generation,
  and automatic computation of lung coverage, confidence score, and symmetry metrics
- **Docker deployment** — Multi-stage Dockerfile (CPU-only PyTorch) with
  docker-compose stack including PostgreSQL database
- **CI/CD pipelines** — GitHub Actions workflows for linting (ruff), testing (pytest
  with coverage), Docker image build, container registry push (GHCR), and manual
  retraining trigger
- **API key authentication** — SHA-256 hashed key storage in JSON, `X-API-Key` header
  validation via FastAPI dependency, CLI tool for key generation
  (`scripts/create_api_key.py`)
- **Rate limiting** — In-memory sliding-window rate limiter per API key with
  configurable per-minute (10) and per-hour (100) thresholds, `Retry-After` header
  on 429 responses
- **Prediction logging** — SQLite-backed logger recording model name, inference
  latency, confidence scores, lung coverage, and image hashes for every prediction
- **Monitoring dashboard** — Streamlit app with four tabs: Overview (KPIs, model
  distribution, throughput), Performance (latency scatter/histogram, P50/P95/P99
  percentiles), Quality (confidence distribution, coverage trends, low-confidence
  alerts), Model Registry (Dice comparison chart)
- **Training pipeline** — Trainer with Focal + Dice + Boundary combined loss,
  cosine annealing scheduler, early stopping, and checkpoint callbacks
- **Data pipeline** — Dataset loader with albumentations augmentation, train/val
  split, and scripts for data download, preprocessing, and adding new data
- **Evaluation pipeline** — Continuous retraining script with model evaluation
  and registration
- **Configuration management** — Pydantic-settings with environment variable
  overrides, YAML config files, and `.env` support
- **Test suite** — 152 tests covering API endpoints, model architectures, model
  registry, inference pipeline, prediction logger, training losses, dataset loading,
  and configuration management
- **Medical disclaimer** — All API responses include `X-Medical-Disclaimer` header;
  response schemas include disclaimer text
- **DICOM support** — Accept `.dcm` and `.dicom` files alongside PNG/JPEG via
  pydicom
