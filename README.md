# Cash Flow Forecasting in the Dutch Dairy Sector

## Project Overview
This repository contains the full codebase for the master’s thesis:  
**"Cash Flow Forecasting in the Dutch Dairy Sector Using Traditional and Machine Learning Models: A Comparative Evaluation of SARIMAX, XGBoost, GRU, and LSTM Models on Farm-Level Data."**  

The study evaluates four forecasting models applied to monthly cash flow data from Dutch dairy farms. The goal is to assess their accuracy, robustness, and interpretability across varying financial profiles and under the inclusion of exogenous variables such as weather.

## Models Used
- **SARIMAX**: Baseline statistical model for capturing seasonality and external regressors.  
- **XGBoost**: Tree-based ensemble model for structured tabular data.  
- **GRU (Gated Recurrent Units)**: Recurrent neural network designed to capture temporal dependencies.  
- **LSTM (Long Short-Term Memory)**: Deep learning model suitable for long-term sequence forecasting.

## Key Concepts
- Time series modeling with autoregressive and exogenous components  
- One-month-ahead forecasting using 12-month input windows  
- Farm-level temporal holdout strategy  
- Clustering to explore performance across heterogeneous subgroups  
- Model interpretability using SHAP and permutation feature importance

## Libraries and Tools

**Data Processing**  
- `pandas`, `numpy`, `scikit-learn`

**Statistical Modeling**  
- `statsmodels`

**Machine Learning**  
- `xgboost`, `scikit-learn`

**Deep Learning**  
- `TensorFlow`, `Keras`

**Interpretability**  
- `SHAP`, custom permutation importance

**Visualization**  
- `matplotlib`, `seaborn`

**Hyperparameter Optimization**  
- `Optuna` for Bayesian optimization of deep learning models  
- Manual `GridSearch` for tuning XGBoost
  
## Notes
- The dataset is **not included** due to confidentiality agreements.
- Models were trained on farm-level monthly data spanning 2020–2024.
- The pipeline includes preprocessing, feature engineering, model training, evaluation, and interpretability analysis.


