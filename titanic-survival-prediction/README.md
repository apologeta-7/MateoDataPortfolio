# 🚢 Titanic Survival Prediction – Kaggle Challenge

## 🎯 Project Overview
This project is part of the **Titanic: Machine Learning from Disaster** challenge on [Kaggle](https://www.kaggle.com/c/titanic).  
The goal is to **predict passenger survival** using real Titanic passenger data and to demonstrate a complete, professional data science workflow — from exploration to model optimization.

This notebook was developed as a **learning-by-doing project**, emphasizing methodology, reasoning, and interpretability over automation.

---

## 🧩 Workflow Summary

### 1️⃣ Data Exploration & Cleaning (EDA)
- Analyzed dataset structure, missing values, and data types.  
- Imputed missing values using:
  - **Mode** for categorical features (`Embarked`)
  - **Median** (grouped by `Sex` and `Pclass`) for `Age`
- Removed redundant columns and created clean, consistent train/test datasets.

### 2️⃣ Feature Engineering
- Extracted and created new features:
  - **Title** from passenger names (Mr, Miss, Master, etc.)
  - **FamilySize** = `SibSp + Parch + 1`
  - **IsAlone** (binary variable)
  - **HasCabin** and **TicketPrefix** from text columns
- Encoded categorical variables and normalized rare classes.

### 3️⃣ Model Building & Comparison
The project includes a **step-by-step model comparison** to understand performance trade-offs and interpretability.

| Model | Accuracy | Key Observations |
|--------|-----------|------------------|
| **Logistic Regression** | 0.8101 | Strong baseline — captures linear relationships (sex, class, age). |
| **Random Forest (limited depth)** | 0.8045 | Excellent generalization and stability. |
| **Random Forest (balanced classes)** | 0.7933 | Slight drop in accuracy, but higher recall for survivors. |
| **XGBoost** | 0.8045 | Matches Random Forest performance with finer control of bias/variance. |

---

## 📈 Results & Insights

- **Best Accuracy:** 0.8101 (Logistic Regression)
- **Best Recall for Survivors:** 0.78 (Balanced Random Forest)
- The Titanic dataset shows **highly linear relationships**, so complex models (Random Forest, XGBoost) didn’t outperform simpler baselines.
- Focused on **model fairness** and **recall improvement** — crucial for minority-class detection.

---

## 💡 Key Learnings

| Category | Takeaway |
|-----------|-----------|
| **Technical** | Practiced feature engineering, model evaluation, and hyperparameter tuning. |
| **Analytical** | Learned how to interpret metrics (precision, recall, F1) and choose models beyond accuracy. |
| **Scientific** | Validated hypotheses through exploratory and quantitative reasoning. |
| **Ethical** | Improved class balance and fairness — prioritizing survivor detection over minor accuracy gains. |

---

## ⚙️ Tech Stack

- **Python 3.12**
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`
- **Environment:** Google Colab / Jupyter Notebook
- **Tools:** `matplotlib`, `seaborn` (for optional visualization)

---

## 📦 Files Included
| File | Description |
|------|-------------|
| `titanic_survival_prediction.ipynb` | Main notebook (EDA, modeling, comparison) |
| `submission_logistic.csv` | Final Kaggle submission (Logistic Regression) |
| `requirements.txt` | Minimal environment to reproduce |
| `.gitignore` | Excludes data/artefacts from the repo |

---

## 🧠 Author
**Mateo Pascual Esseiva**  
📍 Spain · Open to remote  
💼 Data Analyst | MSc Data Science & AI (Nuclio Digital School)  
📧 [m.pascual.ess@gmail.com](mailto:m.pascual.ess@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/mateopascualesseiva/)  
🐙 [GitHub](https://github.com/apologeta-7)

---

## ✨ Notes
This project was built as part of a **self-directed professional learning process**,  
with **AI-assisted mentoring focused on comprehension and best practices** —  
not as auto-generated code, but as a guided, hands-on exploration.

> “A model can detect patterns, but only the human mind can interpret meaning.”  
> — **Mateo Pascual Esseiva**
