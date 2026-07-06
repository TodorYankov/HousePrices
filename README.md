# House Prices - Advanced Regression Techniques

Final Exam Project (Retake) in Machine Learning.

**Competition:** [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
**Repository:** [github.com/TodorYankov/HousePrices](https://github.com/TodorYankov/HousePrices)
**Final Kaggle submission:** `submission_house_prices_final.csv`, public RMSLE **0.12500** (see Section 15.1 in the notebook for discussion of the CV–leaderboard gap)

**Objective:** Predict house sale prices using a stacked ensemble of regression models, with a strong emphasis on rigorous, honest model validation.

**🏆 Final Result:** Honest cross-validation RMSLE **0.11594** (± 0.00880); public Kaggle RMSLE **0.12500** (see notes on the CV–leaderboard gap below).

## ⚠️ Retake Note: Methodological Correction

This retake corrects a bug discovered in the original submission: the previously reported cross-validation RMSLE of **0.11970** was computed **in-sample** (the meta-learner was evaluated on the same data it was fitted on) rather than via genuine out-of-fold cross-validation. This artificially inflated apparent model quality and masked the effect of two high-leverage outliers in the training data.

The notebook (`house_prices_analysis.ipynb`, Sections 13–15.1) documents the full diagnosis and correction process:
1. **Bug identification**: in-sample vs. out-of-fold evaluation produced very different results (0.11970 vs. 0.13325).
2. **Root cause diagnosis**: fold-by-fold breakdown revealed one fold with RMSLE 0.179 vs. ~0.11–0.14 for the others.
3. **Fix**: removal of two well-documented Ames Housing outliers (`GrLivArea` > 4000, low `SalePrice`) reduced the honest CV RMSLE to **0.11594** and stabilized fold-to-fold variance (std 0.0247 → 0.0088).
4. **Follow-up finding**: the corrected model, despite a better internal CV score, scored *worse* on the public Kaggle leaderboard (0.12500) than the original flawed pipeline (0.11974–0.11988) — likely due to distribution mismatch introduced by removing outliers only from the training set. This is analyzed honestly in Section 15.1 rather than hidden.

## 📈 Performance Summary (Honest Cross-Validation, Post-Correction)

| Model | CV RMSLE |
|-------|----------|
| Baseline RF (11 features) | 0.15793 |
| Random Forest (all features, engineered) | 0.13450 |
| Ridge | 0.12006 |
| Gradient Boosting | 0.12001 |
| LightGBM | 0.12541 |
| XGBoost | 0.11723 |
| **Stacking (5 models, Ridge meta, α=0.5)** | **0.11594** |

All values computed after outlier removal (1,458 training rows) using proper out-of-fold cross-validation. Ridge meta-learner regularization (α) was found to have a statistically negligible effect within the tested range (0.01–0.5); α=0.5 was retained as it gave the (nominally) best and most stable result.

## 🧠 Core Approach

Five base models (Random Forest, XGBoost, LightGBM, Gradient Boosting, Ridge) are combined via a Ridge meta-learner using out-of-fold stacking. Feature engineering includes `TotalSF`, `TotalBath`, `HouseAge`, quality-area interaction terms, and neighborhood/seasonal encodings.

## 📂 Historical Experiments

The notebook also documents earlier exploratory work conducted **before** the correction above, retained for transparency:
- Extending the ensemble to 7 base models (CatBoost, ExtraTrees) — did not improve generalization
- Blending with H2O-3 AutoML at various weights (best historical blend: 70/30, RMSLE 0.11921)

These historical results are **not directly comparable** to the corrected pipeline, since they used the original 1,460-row dataset and the flawed in-sample evaluation metric. They are kept for documentation of the exploration process, not as the final reported result.

## 📂 Repository Structure

- `house_prices_analysis.ipynb` – main analysis, including the bug diagnosis and correction (**primary deliverable**)
- `submission_house_prices_final.csv` – final corrected submission (honest CV RMSLE 0.11594, Kaggle RMSLE 0.12500)
- (historical scripts and submissions from earlier exploration, retained for transparency — see notebook Sections 13.2 and 14 for context)

## 🚀 Reproduction

```bash
pip install -r requirements.txt
jupyter notebook house_prices_analysis.ipynb
```

Run all cells in order (Kernel → Restart & Run All). The notebook will:
1. Load and explore the data
2. Remove the two known outliers
3. Engineer features and preprocess
4. Train and evaluate 5 base models + stacking ensemble via honest cross-validation
5. Diagnose and discuss the CV–leaderboard gap
6. Generate `submission_house_prices_final.csv`

## 👤 Author
Todor Yankov
Retake submission: July 2026
Repository: https://github.com/TodorYankov/HousePrices