# Final Report

## 1. Project Scope

This project builds a demand forecasting workflow for Corporacion Favorita store
sales. The goal is to forecast daily sales for each `date x store_nbr x family`
combination across a 16-day horizon.

The work covers data audit, EDA, feature engineering, baseline models, Prophet
comparison, LightGBM modeling, business-oriented evaluation, inference outputs,
API serving, Docker packaging, dashboard exports, presentation assets and tests.

## 2. Data And Features

The project uses the following source tables:

- `train.csv`
- `test.csv`
- `stores.csv`
- `holidays_events.csv`
- `oil.csv`
- `transactions.csv`
- `sample_submission.csv`

The final LightGBM model uses 41 model features, including:

- Calendar features.
- Store metadata.
- Product family.
- Promotion counts.
- Holiday/event features.
- Oil price features.
- Sales lags and shifted rolling means.

The feature contract is saved in
`models/lightgbm/lightgbm_v1_feature_list.json`.

## 3. Validation Design

The model was evaluated with a time-based validation window:

```text
2017-07-31 to 2017-08-15
```

This simulates the 16-day forecasting horizon and avoids random split leakage.
The main operational validation grain is:

```text
date x store_nbr x family
```

## 4. Model Comparison

| Model | Grain | RMSLE | MAE | WAPE | Total bias |
| --- | --- | ---: | ---: | ---: | ---: |
| LightGBM v1 | date-store-family | 0.4167 | 68.14 | 14.6% | -0.08% |
| Rolling mean baseline | date-store-family | 0.5216 | 97.22 | 20.8% | 1.34% |
| LightGBM v1 aggregate | daily total sales | 0.0574 | 36058.20 | 4.3% | -0.08% |
| Prophet benchmark | daily total sales | 0.1070 | 73232.56 | 8.8% | 2.13% |

LightGBM is selected because it improves the best granular baseline, keeps total
bias controlled and supports the operational serving grain.

## 5. Error Analysis

The final evaluation includes error cuts by:

- Date.
- Forecast horizon day.
- Store.
- Family.
- Promotion status.
- Holiday/event status.

These diagnostics are saved in `reports/tables/` and exported for dashboarding
in `dashboard/data_exports/`.

Key business interpretation:

- LightGBM improves WAPE both with and without promotions.
- LightGBM improves WAPE on holiday/event days and normal days.
- Some low-volume families remain unstable because zero sales are common.
- Store and family monitoring should be part of any rollout.

## 6. Inference And Serving

The serving layer is implemented with FastAPI.

Available endpoints:

```text
GET  /health
GET  /health/ready
GET  /api/v1/features
GET  /api/v1/model
POST /api/v1/predict
```

The API validates required artifacts, model-feature compatibility, categorical
alignment, input schema and prediction output. Predictions are inverse
transformed from log scale and clipped to non-negative sales.

The current prediction endpoint expects model-ready feature rows. It is suitable
for batch model serving behind a feature-generation process. A future user-facing
API could add feature generation from raw `date`, `store_nbr` and `family`
inputs.

## 7. Deployment

Docker packaging is included:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

The image copies only the API, source modules, model artifacts and required
serving feature references. Docker health checks call `/health`.

## 8. Tests And CI

The project includes pytest coverage for:

- Required serving artifacts.
- LightGBM feature contract compatibility.
- Forecast service readiness.
- Batch prediction behavior.
- FastAPI endpoints.
- Invalid request handling.

GitHub Actions CI runs:

- Python dependency installation.
- `python -m compileall api src`.
- `python -m pytest`.
- `docker compose config`.
- `docker compose build`.

## 9. Portfolio Deliverables

Generated deliverables:

- `README.md`
- `reports/executive_summary.md`
- `reports/final_report.md`
- `dashboard/data_exports/*.csv`
- `dashboard/powerbi/dashboard_spec.md`
- `presentation/store_sales_forecasting_presentation.pptx`
- `presentation/store_sales_forecasting_presentation.pdf`

Power BI Desktop is still needed to create the final `.pbix` file from the CSV
exports.

## 10. Recommended Next Steps

1. Build the `.pbix` report using `dashboard/data_exports/`.
2. Register the final model artifact in MLflow.
3. Add automated feature generation for raw business requests.
4. Add monitoring for high-risk stores, families, promotion periods and
   holiday/event windows.
5. Add a deployment target and rollback notes once a hosting platform is chosen.

