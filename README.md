# 🍽️ Smart Food Waste Prediction & Optimization for a College Canteen

An end-to-end machine learning and decision-support system that forecasts daily menu-item demand in a college canteen and converts predictions into waste-minimizing cooking quantities using category-specific safety buffers.

> **Data Notice:** No real canteen records are used. The dataset is synthetically generated (`SEED = 42`) to model realistic academic calendars, weather–menu interactions, and kitchen preparation rules. Reported savings demonstrate the methodology in a controlled simulation rather than guaranteed real-world savings.

---

## 📋 Overview

Institutional kitchens typically rely on rules of thumb (such as cooking the previous week's average plus an arbitrary margin), causing heavy food waste during exam weeks, vacations, and holidays, as well as unexpected shortages on busy days.

This repository contains:
1. **End-to-End ML Pipeline (`Smart_Food_Waste_Prediction_and_Optimization.ipynb`):** Reproducible synthetic data generation, data cleaning, leakage-free feature engineering, chronological train/test splitting, 5-fold `TimeSeriesSplit` cross-validation, model benchmarking, safety-buffer optimization, and explainability analysis.
2. **Interactive Streamlit Dashboard (`app.py`):** A 3-tab web application for live morning kitchen demand forecasting, test-period waste/cost simulation analysis, and model evaluation.

---

## 📊 Key Results (Simulated Test Period: 282 Days)

| Model | CV RMSE | Test MAE | Test RMSE | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Selected)** | **39.73** | **24.23** | **38.08** | **0.975** |
| Random Forest | 41.00 | 26.20 | 41.69 | 0.969 |
| Gradient Boosting | 41.43 | 26.01 | 39.96 | 0.972 |
| Decision Tree | 47.86 | 31.40 | 52.54 | 0.951 |
| Linear Regression | 84.45 | 32.70 | 53.31 | 0.950 |
| Naive Baseline (7-Serving Avg) | 154.93 | 110.27 | 157.54 | 0.564 |

### Waste & Shortage Optimization (90% Target Service Level)
Using category-specific safety buffers learned from out-of-fold validation errors (**+12.4%** Breakfast, **+13.6%** Lunch, **+14.3%** Snack, **+16.8%** Beverage):
* **Leftover Food Reduction:** Dropped from **8
