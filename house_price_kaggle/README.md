# 🏠 House Prices – Advanced Regression | Mateo Pascual

A clean, modular, and fully reproducible solution to the **Kaggle “House Prices: Advanced Regression Techniques”** challenge.  
This repository is structured as a professional portfolio project and demonstrates **EDA, data cleaning, robust preprocessing, cross-validated modeling, and final submission**.

---

## 🧭 Project Overview

**Goal:** predict house sale prices from tabular data (numerical + categorical).  
**Approach:** end-to-end pipeline in scikit-learn + LightGBM, with explicit handling of missing data, outliers, and skewed target distributions.

**Highlights**
- Context-aware imputation (`None`/median/mode) with zero leakage (inside Pipeline).
- Percentile-based winsorization (P1–P99) for selected numeric features.
- Target log-transform (`log1p`) to stabilize variance and improve RMSE.
- Reproducible **5-Fold CV** with Ridge vs. LightGBM.
- Final inverse transform (`expm1`) and `submission.csv`.

---

## 📁 Repository Structure

kaggle-house-prices-mateo+++
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│ └─ README_DATA.md
├─ notebooks/
│ ├─ 01_EDA_and_Cleaning.ipynb
│ ├─ 02_Modeling_and_Validation.ipynb
│ └─ 03_Prediction_Submission.ipynb
├─ src/
│ ├─ prep.py
│ └─ model.py
├─ results/
│ ├─ cv_scores.json
│ └─ figs/
└─ submission/
└─ submission.csv

markdown
Copiar código

- **notebooks/**: polished notebooks for EDA, modeling, and inference (in order).
- **src/**: reusable code for preprocessing and modeling.
- **results/**: cross-validation metrics and figures.
- **submission/**: final Kaggle submission file.
- **data/**: instructions to obtain the dataset (no raw CSVs in Git).

---

## ⚙️ Methodology

1. **EDA & Cleaning**
   - Drop high-missing columns: `PoolQC`, `Alley`, `Fence`, `MiscFeature`.
   - Inspect distributions and target skew.

2. **Imputation (context-aware)**
   - *Absence semantics*: `Garage*`, `Bsmt*`, `FireplaceQu`, `MasVnrType` → `"None"`.
   - *Numeric partners*: `GarageArea`, `TotalBsmtSF`, `MasVnrArea`, etc. → `0`.
   - `LotFrontage` → **median per `Neighborhood`**.
   - `Electrical` → global mode.

3. **Outliers**
   - Winsorization (clip to **P1–P99**) on `GrLivArea`, `LotArea`, `TotalBsmtSF`.

4. **Target Transform**
   - `SalePrice_log = log1p(SalePrice)`; prediction inverse via `expm1()`.

5. **Pipeline & Models**
   - `ColumnTransformer`: numeric (median + StandardScaler), categorical (mode + OneHotEncoder).
   - Models: **Ridge** and **LightGBM**.
   - Validation: **KFold(5)**, metric **RMSE (log scale)**.

6. **Tuning (light)**
   - Grid search over `num_leaves`, `learning_rate`, `n_estimators`.

---

## 📊 Results

| Model | RMSE (log) | Notes |
|------|-------------|-------|
| Ridge | ~0.1400 | Strong linear baseline |
| LightGBM (baseline) | ~0.1368 | Captures non-linear interactions |
| LightGBM (tuned) | ~0.1371 | Confirms stable baseline |

> **Interpretation:** RMSE_log ≈ 0.14 → ~15% relative error on price (≈ ±$27k around a $180k home).

---

## 🚀 Reproducibility

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Place Kaggle CSVs under /data (see data/README_DATA.md)

# 3) Run notebooks in order:
# notebooks/01_EDA_and_Cleaning.ipynb
# notebooks/02_Modeling_and_Validation.ipynb
# notebooks/03_Prediction_Submission.ipynb
🧰 Tech Stack
Python 3.11 · Pandas · NumPy · Matplotlib · Seaborn · Scikit-Learn · LightGBM

📌 Notes
No raw data is committed to the repo.

All preprocessing occurs inside the scikit-learn Pipeline to prevent leakage.

The same clipping limits learned on train are applied to test for consistency.

👤 Author
Mateo Pascual — Data Analyst / ML Enthusiast
📍 Spain · Open to US/Global opportunities
