# House Prices – Leakage-Free Machine Learning Pipeline

## Project Overview

This project solves the Kaggle **House Prices: Advanced Regression Techniques**
regression problem using a reproducible and leakage-free machine learning
workflow.

The main objective is not only to obtain a competitive RMSLE score, but also
to demonstrate a clear experimental methodology:

- leakage-free preprocessing;
- feature engineering;
- genuine out-of-fold (OOF) validation;
- structural outlier investigation;
- model comparison;
- Optuna hyperparameter optimization;
- ensemble analysis;
- MLflow experiment tracking;
- subgroup reliability analysis;
- comparison between internal OOF validation and external Kaggle performance.

The final selected model is a **50% Ridge + 50% Optuna-tuned XGBoost ensemble**.

---

## Retake Methodological Correction

An important part of this retake project was correcting the validation
methodology used during earlier experimentation.

An earlier version of the project contained an evaluation issue in which
model performance could appear better than genuine out-of-fold performance.
Rather than retaining those optimistic results, the validation pipeline was
redesigned around strict fold-specific preprocessing and genuine OOF
predictions.

The final workflow therefore treats preprocessing, model fitting,
hyperparameter optimization and ensemble selection as parts of the
cross-validation procedure.

The correction also motivated a detailed investigation of two high-leverage
observations. Their eventual removal was based on an explicit structural rule
(`GrLivArea > 4000` and `SalePrice < 300000`) rather than simply deleting
observations with large prediction errors.

Earlier experimental scores are therefore treated as historical results and
are not directly compared with the final leakage-free pipeline.

---

## Dataset

The project uses the Kaggle House Prices dataset.

Training data:

- 1,460 observations;
- 80 input columns;
- target: `SalePrice`.

Test data:

- 1,459 observations;
- 80 input columns.

The target is transformed using:

\[
z = \log(1 + SalePrice)
\]

This allows RMSE in log-price space to correspond to the RMSLE objective used
for model evaluation.

---

## Leakage-Free Validation

Preventing data leakage is a central methodological requirement of this
project.

Numerical and categorical preprocessing is defined using a
`ColumnTransformer`, but the transformations are not fitted on the complete
dataset before cross-validation.

Inside every cross-validation fold:

1. preprocessing is fitted only on the training fold;
2. the model is trained only on the training fold;
3. the validation fold is transformed using parameters learned from the
   training fold;
4. predictions are generated for the untouched validation observations.

This produces genuine **out-of-fold (OOF) predictions**.

The Kaggle test dataset is never used to fit preprocessing parameters,
hyperparameters, or models.

---

## Feature Engineering

The feature-engineering stage adds interpretable housing characteristics,
including:

- `TotalSF`
- `TotalBath`
- `HouseAge`
- `RemodAge`
- `TotalPorchSF`
- `HasGarage`
- `HasBsmt`
- `HasFireplace`
- `HasPool`
- `Qual_TotalSF`
- `Qual_GrLivArea`

The resulting dataset contains **91 features**:

- 48 numerical features;
- 43 categorical features.

Missing values are replaced by zero only when zero has a direct structural
meaning for the engineered quantity.

No means, medians, modes, category frequencies, or other dataset-level
statistics are learned from the complete dataset during feature engineering.

---

## Structural Outlier Analysis

The initial leakage-free Ridge baseline revealed two highly influential
observations.

A transparent structural rule was defined:

\[
GrLivArea > 4000
\quad \land \quad
SalePrice < 300000
\]

This identifies exactly two unusually large houses with comparatively low
sale prices.

Importantly, these observations were **not removed simply because their OOF
prediction errors were large**. The removal criterion is based on an explicit
relationship between raw housing characteristics and sale price.

After applying the structural rule:

- original training observations: **1,460**
- removed observations: **2**
- final training observations: **1,458**

---

## Model Experiments

Several model families were evaluated using the same leakage-free validation
protocol.

| Experiment | OOF RMSLE |
|---|---:|
| Ridge – all observations | 0.148376 |
| Ridge – diagnostic exclusion only | 0.120386 |
| Ridge – retrained after structural outlier rule | 0.115489 |
| XGBoost – baseline leakage-free CV | 0.115370 |
| XGBoost – Optuna tuned leakage-free CV | 0.113803 |
| LightGBM – leakage-free CV | 0.123607 |
| **50/50 Ridge + Optuna XGBoost** | **0.109780** |

---

## Optuna Hyperparameter Optimization

Optuna is used as a controlled proposal mechanism for XGBoost
hyperparameters.

Every Optuna trial repeats the complete leakage-free five-fold validation
procedure. Preprocessing is fitted independently inside each training fold,
so validation data cannot influence learned preprocessing parameters.

The best configuration achieved:

**OOF RMSLE: 0.113803**

Selected parameters:

```text
n_estimators     = 1100
learning_rate    = 0.039524534394525634
max_depth        = 3
min_child_weight = 3
subsample        = 0.7750590251178121
colsample_bytree = 0.6515493505666603
reg_alpha        = 0.1030285029296229
reg_lambda       = 0.6672670420130714
```

This improves on the baseline XGBoost OOF RMSLE of **0.115370**.

---

## Ensemble Selection

Ridge and Optuna-tuned XGBoost produce highly correlated predictions, but
their errors are not identical.

Observed OOF correlations:

- prediction correlation: **0.98524**
- error correlation: **0.83382**

Several ensemble weights were evaluated using OOF predictions.

The best tested combination was:

- Ridge: **50%**
- Optuna-tuned XGBoost: **50%**

The selected ensemble achieved:

**OOF RMSLE = 0.109780**

Fold-level results:

| Fold | RMSLE |
|---|---:|
| 1 | 0.11607 |
| 2 | 0.10544 |
| 3 | 0.11309 |
| 4 | 0.11445 |
| 5 | 0.09887 |

Mean fold RMSLE: **0.10958**

Fold standard deviation: **0.00648**

The blend outperforms both individual models under the same OOF validation
protocol, supporting the conclusion that the models capture complementary
information.

---

## LightGBM Experiment

LightGBM was evaluated as a possible third ensemble component.

Standalone LightGBM achieved:

**OOF RMSLE = 0.123607**

Its OOF errors were strongly correlated with XGBoost. Controlled three-model
experiments showed that the best weight assigned to LightGBM was **0%**.

Therefore LightGBM was rejected from the final ensemble.

This demonstrates an important experimental principle:

> More models do not automatically produce a better ensemble.

---

## Error and Reliability Analysis

The selected ensemble was analyzed beyond its global RMSLE.

Performance was examined by:

- sale-price range;
- `OverallQual`;
- neighborhood;
- individual high-error observations.

The analysis demonstrates that model reliability is not uniform across all
property groups.

RMSLE by price range:

| Price Range | RMSLE |
|---|---:|
| Low | 0.14921 |
| Lower-Middle | 0.09951 |
| Upper-Middle | 0.07700 |
| High | 0.10051 |

The model performs best on the central part of the housing market and is less
reliable for low-priced properties.

The subgroup analysis also shows that rare `OverallQual` categories and
neighborhoods with small sample sizes should be interpreted cautiously.

---

## MLflow Experiment Tracking

MLflow is used only as an experiment ledger.

It records:

- cleaned Ridge OOF RMSLE;
- baseline XGBoost OOF RMSLE;
- Optuna-tuned XGBoost OOF RMSLE;
- selected Ridge/XGBoost ensemble OOF RMSLE;
- best Optuna hyperparameters;
- number of CV folds;
- validation protocol;
- structural outlier rule.

MLflow does not participate in feature engineering, preprocessing,
cross-validation, hyperparameter selection, model fitting, or test-set
inference.

Local MLflow artifacts such as `mlflow.db` and `mlruns/` are excluded from
version control.

---

## Kaggle Generalization Check

The Kaggle Public Leaderboard is treated as an **external generalization
check**, not as an additional training or validation fold.

Final selected model:

- **OOF RMSLE:** 0.10978
- **Kaggle Public Score:** 0.12789
- **Generalization gap:** 0.01811

An earlier 65% Ridge + 35% XGBoost ensemble achieved:

- **OOF RMSLE:** 0.10987
- **Kaggle Public Score:** 0.12575
- **Generalization gap:** 0.01588

Therefore, the configuration with the best internal OOF result did not obtain
the best Public Leaderboard score.

This discrepancy is reported transparently rather than used for retrospective
leaderboard-driven tuning. Hyperparameters and ensemble weights remain
selected from the leakage-free OOF validation procedure rather than repeated
Public Leaderboard feedback.

---

## Final Model

The final selected predictive system is:

**50% Ridge + 50% Optuna-tuned XGBoost**

Both models are fitted on all **1,458 clean training observations** only after
model selection and error analysis are complete.

Predictions are blended in log-price space:

```python
final_test_pred_log = (
    0.5 * ridge_test_pred_log
    + 0.5 * xgb_test_pred_log
)
```

They are then transformed back to the original `SalePrice` scale:

```python
final_test_pred = np.expm1(final_test_pred_log)
```

Final submission:

```text
submissions/submission_leakage_free_ridge_optuna_xgb_50_50.csv
```

---

## Main Notebook

The complete analysis is available in:

```text
house_prices_leakage_free.ipynb
```

The notebook contains the full workflow from data loading and feature
engineering through leakage-free validation, Optuna tuning, model selection,
error analysis, final training, Kaggle submission and external generalization
analysis.

---

## Repository Structure

The main files for the final project are:

- `house_prices_leakage_free.ipynb` – primary notebook containing the complete
  leakage-free machine learning workflow;
- `requirements.txt` – Python dependencies required to reproduce the project;
- `submissions/submission_leakage_free_ridge_optuna_xgb_50_50.csv` – final
  Kaggle submission generated by the selected ensemble;
- `README.md` – project methodology, results and reproducibility documentation;
- `.gitignore` – excludes local environments, MLflow artifacts, caches and
  other non-project files.

Earlier scripts and experimental artifacts in the Git history represent the
development process and should not be interpreted as the final predictive
pipeline.

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- XGBoost
- LightGBM
- Optuna
- MLflow
- Jupyter Notebook
- Git / GitHub

---

## Reproducibility

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

Then open:

```text
house_prices_leakage_free.ipynb
```

and run the notebook from top to bottom.

Random seeds and cross-validation splits are fixed where applicable to improve
reproducibility.

---

## Conclusions

The main findings of the project are:

1. Leakage-free validation substantially improves the credibility of model
   evaluation.
2. Structural outlier analysis significantly improves Ridge performance and
   fold-to-fold stability.
3. Optuna tuning improves XGBoost from **0.115370** to **0.113803** OOF RMSLE.
4. Ridge and Optuna-tuned XGBoost provide complementary predictive information.
5. Their 50/50 ensemble achieves the best internal result:
   **0.109780 OOF RMSLE**.
6. LightGBM does not improve the selected ensemble and is therefore rejected.
7. Reliability varies across price ranges, quality levels and neighborhoods.
8. The best internal OOF model does not necessarily produce the best score on
   a particular Kaggle Public Leaderboard subset.
9. The Kaggle Public Score is treated as external evidence rather than a
   tuning objective.
10. The OOF-to-Public gap is reported explicitly as a limitation of the
    internal validation estimate.

The project prioritizes **reproducibility, leakage prevention, transparent
experimentation and evidence-based model selection** over simply testing a
large number of models.

---

## Author

**Todor Yankov**

Machine Learning Retake Project  
2026
