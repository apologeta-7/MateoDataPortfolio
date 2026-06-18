# Store Sales Forecasting

End-to-end demand forecasting project for the Kaggle Corporacion Favorita store
sales dataset. The project predicts daily sales for each `date x store_nbr x
family` combination over a 16-day horizon and packages the final LightGBM model
behind a FastAPI serving layer.

## Business Goal

Retail replenishment teams need short-term demand signals to reduce stockouts,
avoid excess inventory and plan around promotions and holidays. This project
builds a first production-style forecasting workflow for that use case.

## Final Model

The recommended model is `lightgbm_v1`, a global LightGBM model trained across
all store-family series.

| Metric | LightGBM | Reference |
| --- | ---: | ---: |
| Granular RMSLE | 0.4167 | 0.5216 baseline |
| Granular WAPE | 14.6% | 20.8% baseline |
| MAE | 68.14 | 97.22 baseline |
| Total forecast bias | -0.08% | 1.34% baseline |
| Daily aggregate RMSLE | 0.0574 | 0.1070 Prophet |

Validation window: `2017-07-31` to `2017-08-15`.

## What Is Included

- Data audit and cleaning outputs in `reports/data_quality/`.
- EDA outputs in `reports/eda_business_analysis/`.
- Feature tables in `data/features/`.
- Baseline, Prophet and LightGBM artifacts in `models/`.
- Final evaluation tables and charts in `reports/tables/` and `reports/figures/`.
- Batch inference outputs in `data/predictions/` and `reports/inference/`.
- FastAPI serving app in `api/`.
- Docker deployment config in `Dockerfile` and `docker-compose.yml`.
- Dashboard import layer in `dashboard/data_exports/`.
- Portfolio presentation in `presentation/`.
- Pytest suite in `tests/`.
- GitHub Actions CI in `.github/workflows/ci.yml`.

## Run Locally

Install production dependencies:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the API:

```powershell
$env:STORE_SALES_PROJECT_ROOT = "F:/Trabajo/store sales/store-sales-forecasting"
..\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

```text
GET  /health
GET  /health/ready
GET  /api/v1/features
GET  /api/v1/model
POST /api/v1/predict
```

`POST /api/v1/predict` expects model-ready feature records. Use
`GET /api/v1/features` to inspect the required 41-feature contract.

## Docker

```powershell
docker compose up --build
```

The service is exposed at:

```text
http://127.0.0.1:8001
```

## Tests

Install development dependencies:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run tests:

```powershell
..\.venv\Scripts\python.exe -m pytest
```

The tests cover artifact availability, model-feature contract compatibility,
forecast service readiness, API endpoints and invalid prediction requests.

## Regenerate Portfolio Artifacts

```powershell
..\.venv\Scripts\python.exe scripts\build_portfolio_artifacts.py
```

This rebuilds:

- `dashboard/data_exports/*.csv`
- `presentation/store_sales_forecasting_presentation.pptx`
- `presentation/store_sales_forecasting_presentation.pdf`

## Key Limitations

- The API currently serves model-ready feature rows; it does not create all
  historical lag and rolling features from only `date`, `store_nbr` and
  `family`.
- Power BI `.pbix` generation is not automated. The repository provides the
  import-ready CSV layer and dashboard specification.
- The project is a portfolio-grade forecasting system, not a fully automated
  enterprise replenishment platform.

