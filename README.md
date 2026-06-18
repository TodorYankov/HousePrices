# House Prices - Advanced Regression Techniques

Final Exam Project in Machine Learning.

**Objective:** Predict house sale prices, classify expensive/cheap properties, cluster neighborhoods, reduce dimensionality, and track experiments with MLflow.

**🏆 Final Result:** RMSLE **0.11921**, Kaggle ranking **~275 / 5094** (top ~5.4%).

## 📈 Performance Progression (Public Kaggle RMSLE)

| Version | Meta-model | Alpha | Base Models | RMSLE | Δ |
|---------|------------|-------|-------------|-------|---|
| Baseline (improved) | Ridge | 0.5 | 5 (RF, XGB, LGB, GB, Ridge) | 0.11988 | – |
| v_01 | Ridge | 0.4 | 5 | 0.11986 | -0.00002 |
| v_02 | Ridge | 0.3 | 5 | 0.11983 | -0.00003 |
| v_03 | Ridge | 0.2 | 5 | 0.11981 | -0.00002 |
| v_04 | Ridge | 0.1 | 5 | 0.11977 | -0.00004 |
| v_05 | Ridge | 0.05 | 5 | 0.11975 | -0.00002 |
| v_06 | Ridge | 0.02 | 5 | 0.11974 | -0.00001 |
| v_07 | Ridge | 0.01 | 5 | 0.11974 | 0 |
| Ensemble 50/50 (v_07 + H2O) | – | – | 5 + H2O AutoML | 0.11940 | -0.00034 |
| Ensemble 60/40 (v_07 + H2O) | – | – | 5 + H2O AutoML | 0.11925 | -0.00015 |
| Ensemble 80/20 (v_07 + H2O) | – | – | 5 + H2O AutoML | 0.11928 | +0.00003 |
| **Ensemble 70/30 (v_07 + H2O)** | – | – | **5 + H2O AutoML** | **0.11921** | **-0.00004** |

> **Best result:** **70/30** ensemble of the manual stacking model (v_07) and H2O-3 AutoML.

## 🧠 Core Approach

The key improvement came from **systematically decreasing the regularization strength (alpha)** of the Ridge meta-model – from 0.5 down to 0.02 – which progressively reduced the RMSLE.

Meta-model coefficients at alpha=0.02 (v_06):
- Random Forest: -0.462
- XGBoost: 0.145
- LightGBM: 0.704
- Gradient Boosting: 0.591
- Ridge (as base model): 0.013 (almost zero contribution)

### H2O-3 AutoML Experiment and Weight Tuning

As an additional experiment, I tested H2O-3 AutoML (50 models, 32 minutes). Its best model (StackedEnsemble) achieved a CV RMSLE of 0.12609. I combined it with v_07 using different ratios to find the optimal balance:

| Weight (v_07 / H2O) | RMSLE |
|---------------------|-------|
| 50/50 | 0.11940 |
| 60/40 | 0.11925 |
| 80/20 | 0.11928 |
| **70/30** | **0.11921** |

The best ratio proved to be **70/30**, which yielded the final score of **0.11921** and a ranking of **~275/5094**.

## 📂 Repository Structure

- `house_prices_analysis.ipynb` – main analysis and experiments
- `house_prices_model.py` – baseline Random Forest
- `house_prices_xgboost.py` – XGBoost
- `house_prices_lightgbm.py` – LightGBM
- `house_prices_stacking.py` – Stacking (3 models)
- `house_prices_improved.py` – Base Stacking (5 models) – RMSLE 0.11988
- `house_prices_improved_v_01.py` to `v_07.py` – sequential versions with decreasing alpha
- `house_prices_h2o_autoML.py` – H2O-3 AutoML script (15-minute test)
- `house_prices_h2o_full.py` – H2O-3 AutoML full experiment (2 hours)
- `house_prices_h2o_ensemble.py` – creates ensemble of v_07 + H2O with various weights
- `ensemble_weight_tuning.py` – automated weight tuning
- `submission_ensemble_v07_70_h2o_30.csv` – **best submission (RMSLE 0.11921)**
- `README.md` – this file

## 🚀 Reproduction

### Quick Reproduction (Best Standalone Model)

```bash
pip install -r requirements.txt
python house_prices_improved_v_06.py
This generates submission_house_prices_improved_v_06.csv with RMSLE 0.11974.

Reproduction of the Best Ensemble (RMSLE 0.11921)
To reproduce the final ensemble (70/30 v_07 + H2O-3 AutoML), follow these steps:

Run H2O-3 AutoML (takes ~32 minutes):

bash
python house_prices_h2o_full.py
This generates submission_h2o_autoML_full.csv.

Generate ensembles with different weights:

bash
python ensemble_weight_tuning.py
This creates multiple CSV files with various ratios (80/20, 70/30, 60/40, 50/50, etc.).

Upload all generated CSV files to Kaggle and select the one with the lowest RMSLE. The best result is achieved with the 70/30 ratio (submission_ensemble_v07_70_h2o_30.csv).

Ensemble requirements: H2O-3 and Java (see requirements.txt).

📊 MLflow
All experiments are tracked and can be explored with:

bash
mlflow ui
👤 Author
Todor Yankov
Date: 18 June 2026
