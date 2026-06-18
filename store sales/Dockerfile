FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STORE_SALES_PROJECT_ROOT=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY src ./src

COPY models/lightgbm/lightgbm_v1_model.txt ./models/lightgbm/
COPY models/lightgbm/lightgbm_v1_config.json ./models/lightgbm/
COPY models/lightgbm/lightgbm_v1_feature_list.json ./models/lightgbm/
COPY models/lightgbm/lightgbm_v1_experiment_summary.json ./models/lightgbm/
COPY models/lightgbm/lightgbm_v1_full_model.txt ./models/lightgbm/
COPY models/lightgbm/lightgbm_v1_full_config.json ./models/lightgbm/

COPY data/features/baseline_test_features.parquet ./data/features/

COPY reports/tables/final_model_recommendation.csv ./reports/tables/

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
