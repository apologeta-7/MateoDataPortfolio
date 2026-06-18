# Business Problem

## Project Context

Corporación Favorita is a large retail company that operates multiple stores and sells different product families across several locations.

In a retail environment, one of the most important operational challenges is deciding how much inventory should be available in each store. If demand is underestimated, stores may face stockouts and lost sales. If demand is overestimated, the company may increase holding costs, waste, and inefficient inventory allocation.

This project addresses that challenge by building a demand forecasting system using historical store sales data.

## Business Problem

The business problem is to support store replenishment decisions by forecasting future sales for each store and product family.

The goal is to predict expected sales for the next 16 days, allowing the business to anticipate demand and make better inventory planning decisions.

The forecasting problem is defined at the following level of granularity:

```text
date × store_nbr × family

This means that the model will estimate the expected sales for each product family in each store for each future date.

Business Objective

The main business objective is to improve store replenishment planning by using historical sales patterns and available business context.

The forecast should help the business answer questions such as:

How much demand should we expect in each store over the next 16 days?
Which product families are likely to require more inventory?
Which stores may face higher demand in the short term?
Where is there a higher risk of understocking or overstocking?
How do promotions, holidays, and store characteristics affect expected demand?
Forecasting Horizon

The project focuses on a 16-day forecasting horizon.

This horizon is aligned with the operational need to plan short-term replenishment and with the structure of the Kaggle Store Sales forecasting task.

The model is expected to generate daily forecasts for each store and product family combination.

Prediction Unit

The prediction unit is:

date × store_nbr × family

Where:

date represents the day being forecasted.
store_nbr identifies the store.
family represents the product family.
sales is the target variable to predict.
Data Sources

The project uses historical data provided by the stores and related operational tables.

The main datasets include:

Dataset	Business Use
train.csv	Historical sales used to train and validate the forecasting models
test.csv	Future dates and store-family combinations to forecast
stores.csv	Store metadata such as city, state, type, and cluster
holidays_events.csv	Holidays and special events that may affect demand
oil.csv	Oil price data used as an external macroeconomic signal
transactions.csv	Historical store transaction volume used as an additional demand signal
sample_submission.csv	Required output format for Kaggle submissions
Intended Users

The intended users of this forecasting system are:

Store replenishment teams
Supply chain analysts
Operations managers
Business analysts
Data science teams supporting retail planning

The forecast is not intended to fully automate inventory decisions. Instead, it should provide a data-driven input to support operational planning.

Business Value

A reliable demand forecasting system can create value by:

Reducing stockouts
Reducing excess inventory
Improving replenishment planning
Helping prioritize high-risk stores and product families
Supporting promotion and holiday planning
Providing visibility into future demand patterns
Main Business Risks

The main business risks associated with inaccurate forecasts are:

Risk	Business Impact
Underforecasting	Stockouts, lost sales, poor customer experience
Overforecasting	Excess inventory, higher storage costs, potential waste
Poor holiday adjustment	Incorrect planning around special dates
Poor promotion adjustment	Misallocation of stock during promotional periods
Store-level errors	Inefficient distribution across locations

For this reason, the project will evaluate not only technical accuracy but also business-oriented error patterns.

Success Criteria

The project will be considered successful if it produces:

A clean and reproducible forecasting pipeline.
Baseline models that provide a fair performance reference.
A LightGBM model that improves over the baselines.
A Prophet comparison for selected time series.
Business-oriented evaluation metrics.
A dashboard to communicate sales and forecast performance.
A basic API to serve predictions.
A final presentation and README suitable for a professional portfolio.
Evaluation Perspective

The model will be evaluated from both technical and business perspectives.

Technical metrics include:

RMSLE
MAE
RMSE
WAPE

Business metrics include:

Forecast bias
Underforecast rate
Overforecast rate
Error by store
Error by product family

This dual evaluation is important because a model can perform well on an aggregate metric while still creating operational risk for specific stores or product families.

Project Scope

This project focuses on building a professional end-to-end forecasting workflow for portfolio purposes.

The scope includes:

Data audit and cleaning
Exploratory data analysis
Feature engineering
Baseline forecasting models
LightGBM as the main forecasting model
Prophet as a comparison model
MLflow experiment tracking
FastAPI model serving
Docker-based deployment
Power BI dashboard
Executive presentation
Final portfolio README

The project does not aim to build a fully automated enterprise replenishment system. Instead, it simulates a realistic first version of a retail demand forecasting solution that could later be extended into production.
```
