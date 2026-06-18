# Power BI Dashboard Specification

This folder contains the dashboard design handoff. The actual `.pbix` file must
be created in Power BI Desktop using the CSV layer in `dashboard/data_exports/`.

## Data Sources

Import these CSV files:

- `dashboard/data_exports/executive_kpis.csv`
- `dashboard/data_exports/model_leaderboard.csv`
- `dashboard/data_exports/future_daily_forecast.csv`
- `dashboard/data_exports/error_by_store.csv`
- `dashboard/data_exports/top_risk_stores.csv`
- `dashboard/data_exports/error_by_family.csv`
- `dashboard/data_exports/top_risk_families.csv`
- `dashboard/data_exports/error_by_date.csv`
- `dashboard/data_exports/promotion_holiday_segments.csv`
- `dashboard/data_exports/feature_importance_top20.csv`

## Page 1: Executive Overview

Purpose: summarize whether the model is useful for short-term replenishment.

Visuals:

- KPI cards: LightGBM RMSLE, LightGBM WAPE, total bias, future predicted sales.
- Bar chart: LightGBM versus baseline RMSLE and WAPE.
- Line chart: future daily predicted sales.
- Text box: recommended model and horizon.

## Page 2: Forecast Validation

Purpose: review validation performance over the 16-day window.

Visuals:

- Date line chart: actual versus predicted total sales.
- Date table: RMSLE, WAPE and total bias by date.
- Conditional formatting on worst WAPE dates.

## Page 3: Store Risk

Purpose: identify stores needing planner review.

Visuals:

- Bar chart: top stores by `wape_lightgbm`.
- Bar chart: top stores by `error_contribution_pct_lightgbm`.
- Slicers: city, state, store type, volume segment.
- Detail table: actual sales, predicted sales, WAPE, bias.

## Page 4: Family Risk

Purpose: identify product families with higher forecast risk.

Visuals:

- Bar chart: top families by WAPE.
- Scatter plot: zero sales percentage versus WAPE.
- Slicers: family and volume segment.
- Detail table: promotion rows percentage, bias, error contribution.

## Page 5: Promotion And Holiday Segments

Purpose: show whether forecast quality changes under special business context.

Visuals:

- Clustered bar: WAPE by promotion/holiday segment.
- KPI cards: event-day WAPE and promotion-day WAPE.
- Detail table: total actual, total predicted, bias and error contribution.

## Page 6: Model Explainability

Purpose: explain the strongest model signals.

Visuals:

- Horizontal bar chart: top 20 features by gain.
- Matrix: gain rank, split rank and feature group.

## Suggested Measures

```DAX
WAPE = DIVIDE(SUM('error_by_store'[total_absolute_error_lightgbm]), SUM('error_by_store'[actual_total_sales_lightgbm]))

Total Bias % = DIVIDE(
    SUM('error_by_store'[predicted_total_sales_lightgbm]) - SUM('error_by_store'[actual_total_sales_lightgbm]),
    SUM('error_by_store'[actual_total_sales_lightgbm])
)

Predicted Sales = SUM('future_daily_forecast'[predicted_sales_sum])
```

