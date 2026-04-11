# Mateo Pascual Esseiva — Data Analytics Portfolio

**Data Analyst | Python · SQL · Power BI | MSc Data Science & AI (Nuclio Digital School)**  
📍 Spain · Open to remote  
🌐 [LinkedIn](https://www.linkedin.com/in/mateopascualesseiva/) · 🐙 [GitHub](https://github.com/apologeta-7)

---

## About

Bilingual Data Analyst with 12+ years of international experience in operations, forecasting, and team management. Currently completing a Master's in Data Science & AI at Nuclio Digital School, with a focus on building production-ready, business-oriented analytics solutions.

Native French speaker. Fluent in English and Spanish.

---

## Projects

### 🏪 DSMarket — Demand Forecasting & Stock Replenishment *(TFM · Nuclio Digital School 2025)*
> End-to-end forecasting system for a supermarket chain (3 cities, 10 stores). Built on the M5 Forecasting dataset (Kaggle). Covers EDA, product clustering, LightGBM forecasting, stock replenishment logic, modular Python pipeline, and a FastAPI endpoint for productization.

**Tech:** Python · LightGBM · scikit-learn · FastAPI · Power BI · Pandas · NumPy  
**Highlights:** 3 LightGBM models evaluated · calibrated E2 selected as operational solution · K-Means product segmentation · Power BI dashboard · REST API design

📁 [View project →](./nuclio-tfm-dsmarket)

---

### 🔒 Malware Risk Prediction *(MSc project · Nuclio Digital School — grade: 8.5/10)*
> Supervised ML model to estimate infection probability for Windows machines using hardware and antivirus configuration features. Business constraint: flag at most 10% of machines (alert budget). 1M-row Kaggle dataset processed by chunks.

**Tech:** Python · scikit-learn · HistGradientBoostingClassifier · RandomizedSearchCV  
**Results:** ROC-AUC 0.715 · Precision @ 10% budget: 0.864

📁 [View project →](https://github.com/apologeta-7/malware_prediction_windows)

---

### 💳 Santander Customer Transaction Prediction *(Kaggle competition)*
> Binary classification to predict whether a customer will make a specific transaction. 200 anonymized numerical features. Key insight: detection and handling of synthetic test rows to build frequency-based features.

**Tech:** Python · LightGBM · Pandas · scikit-learn  
**Results:** ROC-AUC 0.8887 (public leaderboard) · 0.8867 (private)

📁 [View project →](./santander)

---

### 🏠 House Prices — Advanced Regression *(Kaggle challenge)*
> End-to-end regression pipeline with context-aware imputation, outlier winsorization, log-transform of target, and 5-Fold cross-validation. Modular `src/` structure with reusable preprocessing and modeling functions.

**Tech:** Python · scikit-learn · LightGBM · Pandas  
**Results:** RMSE_log ≈ 0.137 (LightGBM) · Ridge baseline 0.140

📁 [View project →](./house_price_kaggle)

---

### 🚢 Titanic Survival Prediction *(Kaggle challenge)*
> Classic binary classification with emphasis on methodology, feature engineering, and model interpretability. Includes Title extraction, FamilySize, IsAlone features. Multiple models compared with focus on fairness and minority-class recall.

**Tech:** Python · scikit-learn · XGBoost · Pandas  
**Results:** Best accuracy 0.8101 (Logistic Regression) · Best survivor recall: Balanced Random Forest

📁 [View project →](./titanic-survival-prediction)

---

## Tech Stack

| Area | Tools |
|------|-------|
| Languages | Python · SQL |
| ML / Forecasting | scikit-learn · LightGBM · XGBoost |
| Data | Pandas · NumPy · PyArrow |
| Visualization | Matplotlib · Seaborn · Power BI |
| API | FastAPI · Pydantic |
| Workflow | Git · Jupyter · Google Colab · VS Code |

---

## Education

**MSc Data Science & AI** — Nuclio Digital School *(2025, ongoing)*  
**Bachillerato** — IES Infanta Elena

---

*All projects include documented notebooks, reproducible pipelines, and results with verifiable metrics.*
