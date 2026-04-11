# Santander Customer Transaction Prediction
### Kaggle Competition · Binary Classification · LightGBM

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-green)
![Kaggle](https://img.shields.io/badge/Kaggle-ROC--AUC%200.8887-orange)

---

## Descripción del proyecto

Proyecto de Machine Learning supervisado basado en la competición oficial de Kaggle del Banco Santander (2019). El objetivo es predecir si un cliente realizará una transacción específica en el futuro, a partir de 200 variables numéricas completamente anonimizadas.

Todo el código ha sido escrito y documentado línea a línea, con el objetivo de entender cada decisión técnica y poder defenderla.

---

## Resultado

| Métrica | Valor |
|---|---|
| ROC-AUC validación | 0.8887 |
| ROC-AUC público Kaggle | 0.8887 |
| ROC-AUC privado Kaggle | 0.8867 |

---

## Estructura del proyecto

```
santander/
├── diario/
│   ├── dia_01.md   # Configuración del entorno y estructura del proyecto
│   ├── dia_02.md   # Revisión de EDAs externos y toma de decisiones
│   └── dia_03.md   # Desarrollo completo del notebook y submission
├── notebooks/
│   └── 01_EDA_Santander.ipynb
├── submission/
│   └── submission_lgbm.csv
└── README.md
```

---

## Decisiones técnicas clave

**1. Detección de filas sintéticas en test**
El análisis exploratorio reveló que el test contiene ~50% de filas duplicadas artificialmente. Identificarlas fue imprescindible para construir features de calidad en el paso siguiente.

**2. Frequency Encoding**
Para cada variable, se calculó cuántas veces aparece cada valor en el test real. Esta técnica amplió el dataset de 200 a 400 features y fue la decisión con mayor impacto en el score final.

**3. LightGBM con scale_pos_weight**
Modelo basado en árboles que captura interacciones no lineales. El desbalance de clases (~9:1) se gestionó con `scale_pos_weight` calculado dinámicamente desde los datos de entrenamiento.

---

## Stack tecnológico

- Python · Pandas · NumPy
- Scikit-learn · LightGBM
- Matplotlib · Seaborn
- Google Colab · VS Code · GitHub

---

## Diario del proyecto

Este proyecto incluye un diario de desarrollo en la carpeta `/diario`. Cada entrada documenta las decisiones tomadas, los problemas encontrados y los aprendizajes del día — una forma honesta de mostrar el proceso real detrás del resultado final.

---

## Siguientes pasos

- Tuning de hiperparámetros con Optuna
- Ensemble con XGBoost y CatBoost
- Cross-validation estratificado (k-fold)
- Análisis SHAP para interpretabilidad individual

---

## Autor

**Mateo Pascual Esseiva**  
📍 Spain · Open to remote  
💼 Data Analyst | MSc Data Science & AI (Nuclio Digital School)  
📧 [m.pascual.ess@gmail.com](mailto:m.pascual.ess@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/mateopascualesseiva/)  
🐙 [GitHub](https://github.com/apologeta-7)
