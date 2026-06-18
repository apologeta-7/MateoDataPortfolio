# Store Sales Forecasting — Project Brief

## 1. Project Context

Corporación Favorita operates multiple stores and sells different product families across several locations.

In retail, one of the most important operational challenges is deciding how much inventory should be available in each store. If demand is underestimated, stores may face stockouts and lost sales. If demand is overestimated, the company may increase holding costs, waste, and inefficient inventory allocation.

This project builds a demand forecasting system using historical store sales data to support short-term replenishment decisions.

## 2. Business Problem

The business problem is to support store replenishment planning by forecasting future demand.

The goal is to predict expected sales for the next 16 days for each store and product family.

The forecasting problem is defined at the following level of granularity:

`date × store_nbr × family`

This means that each prediction estimates the expected sales for one product family in one store on one specific future date.

## 3. Business Objective

The main business objective is to improve store replenishment planning.

The forecast should help the business answer questions such as:

- How much demand should we expect in each store over the next 16 days?
- Which product families are likely to require more inventory?
- Which stores may face higher demand in the short term?
- Where is there a higher risk of understocking or overstocking?
- How do promotions, holidays, and store characteristics affect expected demand?

The forecast is not intended to fully automate inventory decisions. Instead, it provides a data-driven input to support operational planning.

## 4. Forecasting Horizon

The project focuses on a 16-day forecasting horizon.

The model will generate daily forecasts for each store and product family combination over the next 16 days.

This horizon supports short-term replenishment planning and is aligned with the Kaggle Store Sales forecasting task.

## 5. Prediction Unit

The prediction unit is:

`date × store_nbr × family`

Where:

- `date` represents the day being forecasted.
- `store_nbr` identifies the store.
- `family` represents the product family.
- `sales` is the target variable to predict.

## 6. Target Variable

The target variable is:

`sales`

The model will predict daily sales for each store and product family.

Sales values must be non-negative. Any negative model predictions will be clipped to zero before evaluation or submission.

## 7. Data Sources

The project uses historical data provided by the stores and related operational tables.

| Dataset                 | Business / Modeling Use                                                 |
| ----------------------- | ----------------------------------------------------------------------- |
| `train.csv`             | Historical sales used to train and validate the forecasting models      |
| `test.csv`              | Future dates and store-family combinations to forecast                  |
| `stores.csv`            | Store metadata such as city, state, type, and cluster                   |
| `holidays_events.csv`   | Holidays and special events that may affect demand                      |
| `oil.csv`               | Oil price data used as an external macroeconomic signal                 |
| `transactions.csv`      | Historical store transaction volume used as an additional demand signal |
| `sample_submission.csv` | Required output format for Kaggle submissions                           |

## 8. Intended Users

The intended users of this forecasting system are:

- Store replenishment teams
- Supply chain analysts
- Operations managers
- Business analysts
- Data science teams supporting retail planning

## 9. Business Value

A reliable demand forecasting system can create value by:

- Reducing stockouts
- Reducing excess inventory
- Improving replenishment planning
- Helping prioritize high-risk stores and product families
- Supporting promotion and holiday planning
- Providing visibility into future demand patterns

## 10. Main Business Risks

| Risk                      | Business Impact                                         |
| ------------------------- | ------------------------------------------------------- |
| Underforecasting          | Stockouts, lost sales, poor customer experience         |
| Overforecasting           | Excess inventory, higher storage costs, potential waste |
| Poor holiday adjustment   | Incorrect planning around special dates                 |
| Poor promotion adjustment | Misallocation of stock during promotional periods       |
| Store-level errors        | Inefficient distribution across locations               |

For this reason, the project will evaluate not only technical accuracy but also business-oriented error patterns.

## 11. Modeling Problem

This project is framed as a supervised time series forecasting problem.

The model learns from historical sales patterns and related business context to predict future sales.

The modeling granularity is:

`store_nbr × family × date`

This level is appropriate because:

- Store-level demand can vary significantly by location.
- Product families have different demand patterns.
- Calendar effects can change sales behavior by date.
- Promotions and holidays may affect different stores and families differently.

The project will not forecast at individual SKU level because the available dataset does not provide individual product-level information.

## 12. Feature Groups

The model will use several groups of features.

### Calendar Features

Calendar features help the model learn recurring time patterns.

Examples:

- Day of week
- Week of year
- Month
- Year
- Day of month
- Weekend flag
- Month start flag
- Month end flag

These features are safe because future calendar dates are known in advance.

### Store Features

Store features describe each store.

Examples:

- City
- State
- Store type
- Store cluster

These features allow the model to learn differences between stores.

### Product Family Features

The main product feature is:

`family`

This categorical feature helps the model learn different demand patterns across product families.

### Promotion Features

Promotion features come from the `onpromotion` column.

Promotions are important because they can directly affect demand.

The project will treat `onpromotion` as a known future feature because it is available in both the training and test datasets.

### Holiday and Event Features

Holiday and event features will be created from `holidays_events.csv`.

Potential features include:

- National holidays
- Regional holidays
- Local holidays
- Transferred holidays
- Bridge days
- Additional days
- Event type
- Locale type

Transferred holidays must be handled carefully because not all listed holidays should be treated as effective holidays.

### Oil Features

Oil prices may provide useful macroeconomic context.

Potential features include:

- Daily oil price
- Forward-filled oil price
- Oil price change
- Rolling oil price average

Oil data may contain missing values due to weekends or non-trading days. These values should be handled using time-aware imputation methods.

### Transaction Features

Transactions may be useful as a proxy for store traffic.

However, transactions require special care because future transaction values are not available at prediction time.

For this reason, transaction data will be used primarily for:

- Exploratory analysis
- Historical demand understanding
- Potential lag-based features only if they are available before the prediction date

Raw future transaction values must not be used as model inputs because that would create data leakage.

### Lag Features

Lag features use past sales to predict future sales.

Examples:

- Sales lag 1
- Sales lag 7
- Sales lag 14
- Sales lag 28

Lag features must be created using only past data available before the prediction date.

### Rolling Features

Rolling features summarize recent demand behavior.

Examples:

- Rolling mean over 7 days
- Rolling mean over 14 days
- Rolling mean over 28 days
- Rolling standard deviation
- Rolling maximum
- Rolling minimum

Rolling features must be shifted before calculation to avoid leakage.

## 13. Leakage Prevention

Avoiding data leakage is a central requirement of this project.

The model must only use information that would be available at the time of prediction.

Rules:

1. Future sales must never be used as features.
2. Rolling statistics must be shifted before calculation.
3. Lag features must only reference dates before the prediction date.
4. Future transaction values must not be used.
5. Calendar features are allowed because future dates are known.
6. Store metadata is allowed because it is static.
7. Promotion information is allowed because it is available in the test dataset.
8. Holiday information is allowed because holidays are known in advance.
9. Oil prices must be handled carefully. Only values available up to the prediction date should be used in a realistic setting.

## 14. Validation Strategy

The project will use time-based validation.

Random train-test splitting is not appropriate because this is a time series forecasting problem.

The validation approach will simulate a realistic forecasting scenario:

1. Train the model using historical data up to a certain date.
2. Predict the next 16 days.
3. Compare predictions against actual sales.
4. Repeat this process across one or more temporal validation windows.

The validation design should answer:

- Does the model perform better than simple baselines?
- Is the model stable across time?
- Does the model create systematic underforecasting or overforecasting?
- Which stores or families are most difficult to forecast?

## 15. Candidate Models

### Baseline Models

Baseline models will be used as performance references.

They are essential because a machine learning model is only useful if it improves over simple and explainable alternatives.

Planned baselines:

| Baseline                      | Description                                               |
| ----------------------------- | --------------------------------------------------------- |
| Last value baseline           | Uses the most recent known sales value                    |
| Seasonal naive 7-day baseline | Uses sales from the same day of the previous week         |
| Rolling mean baseline         | Uses recent average sales                                 |
| Store-family-weekday average  | Uses historical average by store, family, and day of week |

### LightGBM

LightGBM will be the main model of the project.

It is selected because:

- It performs well on structured tabular data.
- It handles many categorical and numerical features.
- It scales better than training one model per time series.
- It can learn interactions between store, family, calendar, promotion, and lag features.
- It can be serialized and served through an API.
- It is suitable for a professional portfolio project.

Planned LightGBM versions:

| Version        | Description                                     |
| -------------- | ----------------------------------------------- |
| LightGBM v1    | Calendar, store, family, and promotion features |
| LightGBM v2    | Adds lag features                               |
| LightGBM v3    | Adds rolling features                           |
| LightGBM v4    | Adds holidays and oil features                  |
| LightGBM final | Best performing and most defensible version     |

### Prophet

Prophet will be used as a comparison model, not as the main production model.

Prophet is useful for understanding trend and seasonality in selected time series.

However, Prophet is less practical as the main solution for this project because the dataset contains many store-family combinations.

The Prophet comparison will be limited to selected representative series, such as:

- High-volume product families
- High-volume stores
- Stable demand patterns
- Series with clear seasonality

The goal is to compare interpretability and forecasting behavior, not to replace the main LightGBM model.

## 16. Models Not Included in the Initial Scope

The following models are intentionally excluded from the initial project scope:

| Model Type                      | Reason                                                      |
| ------------------------------- | ----------------------------------------------------------- |
| LSTM                            | Adds complexity without clear portfolio value at this stage |
| Transformers                    | Too complex for the first professional version              |
| DeepAR                          | Interesting but not necessary for the first version         |
| AutoML                          | Reduces control over the modeling narrative                 |
| Multiple boosted models at once | May distract from a clean and defendable workflow           |

These models could be explored in future iterations, but the first version will prioritize clarity, reproducibility, and business value.

## 17. Evaluation Metrics

The project will evaluate models from both technical and business perspectives.

### Technical Metrics

| Metric | Purpose                                 |
| ------ | --------------------------------------- |
| RMSLE  | Official Kaggle-style evaluation metric |
| MAE    | Average absolute error in sales units   |
| RMSE   | Penalizes larger errors                 |
| WAPE   | Business-friendly percentage error      |

### Business Metrics

| Metric             | Purpose                                                   |
| ------------------ | --------------------------------------------------------- |
| Bias               | Measures systematic overforecasting or underforecasting   |
| Underforecast Rate | Measures how often the model predicts below actual demand |
| Overforecast Rate  | Measures how often the model predicts above actual demand |
| Error by store     | Identifies problematic stores                             |
| Error by family    | Identifies problematic product families                   |

Business metrics are important because a model can perform well overall while still creating operational problems in specific stores or product families.

## 18. Model Selection Criteria

The final model will not be selected only based on a single metric.

The selected model should satisfy the following criteria:

1. Improve over baseline models.
2. Perform well on RMSLE and WAPE.
3. Avoid extreme systematic bias.
4. Be stable across stores and product families.
5. Be explainable enough for a business audience.
6. Be reproducible from the project pipeline.
7. Be suitable for serving through a basic API.

## 19. Project Deliverables

The project will produce both technical and business-facing deliverables.

### Technical Deliverables

| Deliverable                  | Location                              |
| ---------------------------- | ------------------------------------- |
| Cleaned datasets             | `data/interim/`                       |
| Model-ready datasets         | `data/processed/`                     |
| Feature tables               | `data/features/`                      |
| Model predictions            | `data/predictions/`                   |
| Trained LightGBM model       | `models/lightgbm/`                    |
| Prophet comparison artifacts | `models/prophet/`                     |
| MLflow experiment tracking   | `mlruns/`                             |
| FastAPI application          | `api/`                                |
| Docker configuration         | `Dockerfile` and `docker-compose.yml` |

### Business / Portfolio Deliverables

| Deliverable             | Location                                                 |
| ----------------------- | -------------------------------------------------------- |
| Executive README        | `README.md`                                              |
| Final report            | `reports/final_report.md`                                |
| Executive summary       | `reports/executive_summary.md`                           |
| Power BI dashboard      | `dashboard/powerbi/`                                     |
| Dashboard export tables | `dashboard/data_exports/`                                |
| Final presentation      | `presentation/store_sales_forecasting_presentation.pptx` |
| Presentation PDF        | `presentation/store_sales_forecasting_presentation.pdf`  |

## 20. Project Scope

This project focuses on building a professional end-to-end forecasting workflow for portfolio purposes.

The scope includes:

- Data audit and cleaning
- Exploratory data analysis
- Feature engineering
- Baseline forecasting models
- LightGBM as the main forecasting model
- Prophet as a comparison model
- MLflow experiment tracking
- FastAPI model serving
- Docker-based deployment
- Power BI dashboard
- Executive presentation
- Final portfolio README

The project does not aim to build a fully automated enterprise replenishment system.

Instead, it simulates a realistic first version of a retail demand forecasting solution that could later be extended into production.

## 21. Final Modeling Decision

The expected final model is a LightGBM-based global forecasting model trained across all store-family combinations.

This model will use historical sales, calendar features, store metadata, promotions, holidays, and carefully selected lag and rolling features.

Prophet will be used as an interpretability and comparison benchmark for selected individual time series.

The final objective is to produce a model that is not only accurate, but also useful, explainable, reproducible, and suitable for a professional portfolio presentation.
