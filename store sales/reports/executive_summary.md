# Executive Summary

## Recommendation

Use `lightgbm_v1` as the final forecasting model for the next MLOps phase.
It is the strongest model at the required operational grain:
`date x store_nbr x family`.

## Business Result

The model forecasts 16 days of sales for each store and product family. On the
validation window from `2017-07-31` to `2017-08-15`, LightGBM improved the best
granular baseline:

| Metric | LightGBM | Best baseline |
| --- | ---: | ---: |
| RMSLE | 0.4167 | 0.5216 |
| WAPE | 14.6% | 20.8% |
| MAE | 68.14 | 97.22 |
| Total bias | -0.08% | 1.34% |

At daily aggregate level, LightGBM also outperformed the Prophet benchmark:
`0.0574` RMSLE versus `0.1070`.

## Business Use

The forecast is suitable as a planning input for:

- Short-term replenishment review.
- Store and family demand prioritization.
- Promotion and holiday monitoring.
- Portfolio demonstration of an end-to-end forecasting workflow.

## Main Risk

Some stores, families, promotion periods and holiday/event dates still require
monitoring. The model is strong globally, but operational users should use the
error-by-store, error-by-family and segment diagnostics before trusting every
individual forecast equally.

## Delivery Status

The project now includes:

- Final model and prediction artifacts.
- FastAPI serving layer.
- Docker configuration.
- Dashboard data exports.
- Executive presentation and PDF.
- Automated tests.
- GitHub Actions CI workflow.

The remaining manual handoff item is building the actual `.pbix` file in Power
BI Desktop from `dashboard/data_exports/`.

