# Dashboard Data Exports

These CSV files are generated from the saved model evaluation outputs.
They are intended as the Power BI import layer for the portfolio dashboard.

Recommended Power BI pages:
- Executive overview: `executive_kpis.csv`, `model_leaderboard.csv`.
- Forecast horizon: `future_daily_forecast.csv`.
- Store risk: `error_by_store.csv`, `top_risk_stores.csv`.
- Family risk: `error_by_family.csv`, `top_risk_families.csv`.
- Segment diagnostics: `promotion_holiday_segments.csv`.
- Explainability: `feature_importance_top20.csv`.

Regenerate with:

```powershell
..\.venv\Scripts\python.exe scripts\build_portfolio_artifacts.py
```
