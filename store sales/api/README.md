# Store Sales Forecasting API

FastAPI service for serving the final LightGBM forecasting model.

The API does not run notebooks. It loads the saved model, feature contract, and
supporting artifacts from the project folders.

## Run locally

From the project root:

```powershell
..\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Run with Docker

Build and start the API container:

```powershell
docker compose up --build
```

The container maps the API to:

```text
http://127.0.0.1:8001
```

OpenAPI docs:

```text
http://127.0.0.1:8001/docs
```

## Endpoints

```text
GET  /health
GET  /health/ready
GET  /api/v1/features
GET  /api/v1/model
POST /api/v1/predict
```

## Prediction contract

`POST /api/v1/predict` expects model-ready feature records. Each record must
include the model feature contract returned by `GET /api/v1/features`.

Example shape:

```json
{
  "records": [
    {
      "date": "2017-08-16",
      "store_nbr": 1,
      "family": "AUTOMOTIVE"
    }
  ],
  "include_diagnostics": false
}
```

The example above shows only the identifying fields. A real request must also
include all model features listed by `/api/v1/features`.
