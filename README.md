# House Prices – Leakage-Free Machine Learning Pipeline

Machine Learning project based on the Kaggle competition  
**House Prices: Advanced Regression Techniques**.

## Project Objective

The goal of this project is to predict residential sale prices from the Ames Housing dataset.

The project focuses not only on predictive performance, but also on:

- leakage-free preprocessing;
- honest out-of-fold cross-validation;
- interpretable experiment design;
- error analysis;
- model comparison and selection;
- model limitations;
- generalization to unseen Kaggle data.

---

## Methodology

The final version of the project uses a leakage-free machine learning workflow.

All learned preprocessing operations are performed inside the cross-validation pipeline.

For every cross-validation fold:

1. The preprocessor is fitted only on the training fold.
2. Missing values are imputed using information from the training fold only.
3. Categorical encoding and scaling are learned from the training fold only.
4. The validation fold is transformed using the fitted preprocessor.
5. The model is trained on the training fold.
6. Predictions are generated for the untouched validation fold.

The Kaggle test dataset is never used to fit imputers, encoders, scalers, or models.

This allows the Out-of-Fold (OOF) validation results to provide a more realistic estimate of
generalization performance.

---

## Feature Engineering

Several domain-inspired features are created from the original Ames Housing predictors:

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

These features are created exclusively from predictor variables and do not use the target
`SalePrice`.

After feature engineering:

- Numerical features: **48**
- Categorical features: **43**
- Total features: **91**

---

## Leakage-Free Baseline

The first leakage-free Ridge experiment was evaluated using 5-fold cross-validation.

Results:

| Fold | RMSLE |
|---|---:|
| Fold 1 | 0.13315 |
| Fold 2 | 0.12910 |
| Fold 3 | 0.22172 |
| Fold 4 | 0.12300 |
| Fold 5 | 0.10674 |

Overall results:

- Mean Fold RMSLE: **0.14274**
- Fold Standard Deviation: **0.04050**
- OOF RMSLE: **0.14838**

The unusually poor third fold motivated a detailed OOF error analysis.

---

## Structural Outlier Analysis

The OOF analysis revealed two extreme observations:

- **Id 524**
- **Id 1299**

Both properties had extremely large living areas and high overall quality but unusually low
sale prices.

A transparent structural rule was investigated:

`GrLivArea > 4000 and SalePrice < 300000`

The rule identified exactly these two observations.

Importantly, the observations were not removed simply because the model predicted them poorly.
The removal decision was based on their unusual structural relationship between property size
and sale price.

After applying the structural rule:

- Original observations: **1460**
- Removed observations: **2**
- Final training observations: **1458**

---

## Ridge After Structural Outlier Removal

The complete leakage-free Ridge pipeline was retrained and evaluated again.

Results:

| Fold | RMSLE |
|---|---:|
| Fold 1 | 0.12305 |
| Fold 2 | 0.11048 |
| Fold 3 | 0.11705 |
| Fold 4 | 0.12355 |
| Fold 5 | 0.10185 |

Overall:

- Mean Fold RMSLE: **0.11520**
- Fold Standard Deviation: **0.00819**
- OOF RMSLE: **0.11549**

This was a substantial improvement over the original leakage-free baseline.

---

## Model Comparison

Multiple models were evaluated using the same leakage-free OOF methodology.

| Model | OOF RMSLE |
|---|---:|
| Ridge | **0.11549** |
| XGBoost | **0.11537** |
| LightGBM | 0.12361 |
| Ridge + XGBoost 50/50 | **0.11069** |

XGBoost achieved a slightly better standalone OOF score than Ridge.

However, the prediction and error analysis showed that the two models were sufficiently
different to justify testing an ensemble.

Prediction correlation between Ridge and XGBoost:

**0.98551**

OOF error correlation:

**0.83920**

---

## Ensemble Selection

A controlled OOF blending experiment tested different Ridge and XGBoost weights.

The best tested combination was:

- Ridge: **50%**
- XGBoost: **50%**

OOF RMSLE:

**0.11069**

### Fold-by-Fold Ensemble Performance

| Fold | RMSLE |
|---|---:|
| Fold 1 | 0.11664 |
| Fold 2 | 0.10544 |
| Fold 3 | 0.11515 |
| Fold 4 | 0.11577 |
| Fold 5 | 0.09937 |

Overall:

- Mean Fold RMSLE: **0.11047**
- Fold Standard Deviation: **0.00688**
- OOF RMSLE: **0.11069**

The ensemble improved both individual models while maintaining stable performance across folds.

---

## LightGBM Experiment

LightGBM was also evaluated using the same leakage-free methodology.

LightGBM OOF RMSLE:

**0.12361**

The OOF error correlations were:

| | Ridge | XGBoost | LightGBM |
|---|---:|---:|---:|
| Ridge | 1.000 | 0.839 | 0.786 |
| XGBoost | 0.839 | 1.000 | 0.922 |
| LightGBM | 0.786 | 0.922 | 1.000 |

A controlled three-model blending experiment tested increasing LightGBM contributions.

The optimal tested combination was:

- Ridge: **50%**
- XGBoost: **50%**
- LightGBM: **0%**

Therefore, LightGBM was excluded from the final model.

Adding it increased complexity without improving OOF performance.

---

## Final Model

The selected final ensemble is:

**50% Ridge + 50% XGBoost**

Final leakage-free validation performance:

**OOF RMSLE = 0.11069**

The final models were trained using **1458 training observations** and predictions were generated
for **1459 Kaggle test observations**.

---

## Error Analysis

The final ensemble produced:

- Mean residual: **0.00071**
- Median residual: **0.00395**
- Mean absolute log error: **0.07482**
- Median absolute log error: **0.05283**

The analysis showed that model reliability is not uniform across the housing market.

### Performance by Price Range

| Price Segment | RMSLE |
|---|---:|
| Low | 0.15106 |
| Lower-Middle | 0.09920 |
| Upper-Middle | **0.07768** |
| High | 0.10152 |

The model performs best in the middle of the price distribution.

Low-priced properties are substantially more difficult to predict.

---

## Performance by Neighborhood

Model performance also varies considerably across neighborhoods.

Examples:

| Neighborhood | RMSLE |
|---|---:|
| IDOTRR | 0.24541 |
| OldTown | 0.14567 |
| Edwards | 0.12717 |
| NridgHt | 0.10044 |
| NAmes | 0.09615 |
| Gilbert | 0.07403 |
| CollgCr | 0.06488 |

These results demonstrate that a single global RMSLE score does not describe model reliability
equally well for all types of properties.

Small neighborhood groups should also be interpreted cautiously because their estimates are
based on relatively few observations.

---

## Prediction Sanity Checks

Before creating the final submission, the predictions were checked for invalid values.

Results:

- Number of predictions: **1459**
- NaN predictions: **0**
- Infinite predictions: **0**
- Negative predictions: **0**

The highest and lowest predicted properties were also inspected manually.

High predictions generally corresponded to properties with:

- high `OverallQual`;
- large `GrLivArea`;
- large total floor area;
- newer construction;
- multiple garage spaces.

Low predictions generally corresponded to:

- low `OverallQual`;
- smaller living areas;
- older construction;
- limited or no garage capacity.

No manual clipping or correction of individual predictions was applied.

---

## Kaggle Generalization Check

The final submission was evaluated on the Kaggle public leaderboard.

| Evaluation | RMSLE |
|---|---:|
| Leakage-Free OOF | **0.11069** |
| Kaggle Public Score | **0.12808** |
| Generalization Gap | **0.01739** |

The public Kaggle score is worse than the internal OOF estimate.

This indicates that the cross-validation estimate was somewhat optimistic relative to unseen
competition data.

The gap is reported explicitly rather than hidden.

The public leaderboard was treated as an external generalization check rather than as another
validation dataset.

Repeated tuning solely against the public leaderboard could lead to leaderboard overfitting.

---

## Final Submission

The final Kaggle submission is:

`submissions/submission_leakage_free_ridge_xgb_50_50.csv`

Final ensemble:

**50% Ridge + 50% XGBoost**

Kaggle Public Score:

**0.12808**

---

## Repository Structure

```text
HousePrices/
│
├── house_prices_leakage_free.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── submissions/
    └── submission_leakage_free_ridge_xgb_50_50.csv
```

The Kaggle dataset is expected locally in the `data/` directory:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

The dataset is not stored in the Git repository.

---

## Reproduction

### Installation

Create or activate a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The project was developed using Python 3.12.

### Running the Project

Open:

`house_prices_leakage_free.ipynb`

Run the notebook cells sequentially.

The notebook performs:

1. Raw data loading
2. Feature and target separation
3. Leakage-safe feature engineering
4. Leakage-free preprocessing
5. Five-fold cross-validation
6. Genuine OOF prediction generation
7. OOF error analysis
8. Structural outlier investigation
9. Ridge evaluation
10. XGBoost evaluation
11. OOF model blending
12. LightGBM evaluation
13. Final model selection
14. Detailed error analysis
15. Final model training
16. Kaggle prediction generation
17. Prediction sanity checks
18. OOF vs Kaggle generalization analysis

---

## Final Conclusion

The main objective of the final version is not simply to obtain the lowest possible leaderboard score.

The project follows a clear experimental workflow:

**hypothesis → experiment → OOF evaluation → error analysis → decision**

The Ridge/XGBoost ensemble was selected because the OOF evidence supported the combination.

LightGBM was tested but rejected because it did not improve validation performance.

The final analysis also documents where the model performs well and where its predictions are less reliable.

The final model should therefore be viewed as a validated predictive system with documented limitations rather than only as a Kaggle leaderboard score.

---

## Author

**Todor Yankov**

Machine Learning Retake Project  
2026   