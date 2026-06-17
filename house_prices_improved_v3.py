# house_prices_improved_v2.py
"""
Weighted Average Ensemble with ExtraTrees
Based on best model (0.11988) – replaces stacking meta-learner with optimized weights
Includes: RF, XGB, LGB, GB, Ridge, ExtraTrees
Target RMSLE: 0.116-0.118
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
from scipy.optimize import minimize
import warnings
import optuna
import mlflow

warnings.filterwarnings('ignore')

print("=" * 80)
print("🏆 WEIGHTED ENSEMBLE WITH EXTRATREES – target RMSLE < 0.118")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")


# ============================================================================
# 2. MISSING VALUE HANDLING (as in your best model)
# ============================================================================
def fix_missing_values(df):
    data = df.copy()
    num_features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1',
                    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
                    'BsmtHalfBath', 'GarageArea', 'GarageCars']
    for col in num_features:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    cat_features = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',
                    'MasVnrType', 'Electrical', 'KitchenQual', 'SaleType',
                    'Functional', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish',
                    'GarageQual', 'GarageCond', 'FireplaceQu', 'PoolQC', 'Fence',
                    'MiscFeature', 'Alley']
    for col in cat_features:
        if col in data.columns:
            data[col] = data[col].fillna('None')
    return data


train = fix_missing_values(train)
test = fix_missing_values(test)


# ============================================================================
# 3. FEATURE ENGINEERING (all proven features from your best model)
# ============================================================================
def add_advanced_features(df, neigh_median=None):
    data = df.copy()
    data['TotalSF'] = data['TotalBsmtSF'] + data['GrLivArea']
    data['TotalSF_log'] = np.log1p(data['TotalSF'])
    data['TotalBath'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']

    if 'OverallQual' in data.columns and 'OverallCond' in data.columns:
        data['OverallScore'] = data['OverallQual'] * data['OverallCond']
        data['OverallQual_sq'] = data['OverallQual'] ** 2

    if 'YearBuilt' in data.columns:
        data['HouseAge'] = 2025 - data['YearBuilt']
        data['HouseAge_sq'] = data['HouseAge'] ** 2
        data['log_LotArea'] = np.log1p(data['LotArea'])
        data['log_GrLivArea'] = np.log1p(data['GrLivArea'])

    if 'YearRemodAdd' in data.columns:
        data['YearsSinceRemod'] = 2025 - data['YearRemodAdd']
        data['RemodAge'] = (data['YearRemodAdd'] - data['YearBuilt']).clip(0, 50)

    # Ratios
    data['AreaPerRoom'] = data['GrLivArea'] / (data['TotRmsAbvGrd'] + 1)
    data['BsmtRatio'] = data['TotalBsmtSF'] / (data['GrLivArea'] + 1)
    data['GarageRatio'] = data['GarageArea'] / (data['GrLivArea'] + 1)

    # Quality numeric
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    if 'KitchenQual' in data.columns:
        data['KitchenQual_num'] = data['KitchenQual'].map(qual_map).fillna(0)
    if 'BsmtQual' in data.columns:
        data['BsmtQual_num'] = data['BsmtQual'].map(qual_map).fillna(0)
    if 'ExterQual' in data.columns:
        data['ExterQual_num'] = data['ExterQual'].map(qual_map).fillna(0)
    if 'FireplaceQu' in data.columns:
        data['FireplaceQu_num'] = data['FireplaceQu'].map(qual_map).fillna(0)

    # Interactions
    data['Qual_Area'] = data['OverallQual'] * data['GrLivArea']
    data['Qual_TotalSF'] = data['OverallQual'] * data['TotalSF']
    data['Age_Qual'] = data['HouseAge'] * data['OverallQual']
    data['Qual_PerAge'] = data['OverallQual'] / (data['HouseAge'] + 1)

    # Neighborhood median (if training and SalePrice exists)
    if 'SalePrice' in data.columns:
        neigh_median = data.groupby('Neighborhood')['SalePrice'].median().to_dict()
        data['NeighborhoodMedian'] = data['Neighborhood'].map(neigh_median)
    elif neigh_median is not None:
        data['NeighborhoodMedian'] = data['Neighborhood'].map(neigh_median)
        data['NeighborhoodMedian'] = data['NeighborhoodMedian'].fillna(180000)

    # Top neighborhoods
    top_neigh = ['StoneBr', 'NridgHt', 'NoRidge', 'Somerst']
    data['TopNeighborhood'] = data['Neighborhood'].isin(top_neigh).astype(int)

    # Seasonality
    if 'MoSold' in data.columns:
        data['Spring'] = data['MoSold'].isin([3, 4, 5]).astype(int)
        data['Summer'] = data['MoSold'].isin([6, 7, 8]).astype(int)
        data['Fall'] = data['MoSold'].isin([9, 10, 11]).astype(int)
        data['Winter'] = data['MoSold'].isin([12, 1, 2]).astype(int)

    # Expensive material
    expensive = ['BrkFace', 'Stone', 'BrkComm', 'CBlock']
    if 'Exterior1st' in data.columns:
        data['ExpensiveMaterial'] = data['Exterior1st'].isin(expensive).astype(int)

    # Total porch
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    data['TotalPorchSF'] = sum([data[col].fillna(0) for col in porch_cols if col in data.columns])
    data['HasPorch'] = (data['TotalPorchSF'] > 0).astype(int)

    return data, neigh_median


train, neigh_median_dict = add_advanced_features(train)
test, _ = add_advanced_features(test, neigh_median=neigh_median_dict)

# ============================================================================
# 4. ENCODE CATEGORICALS
# ============================================================================
cat_cols = train.select_dtypes(include=['object']).columns
print(f"Categorical columns: {len(cat_cols)}")
for col in cat_cols:
    if col != 'SalePrice':
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

# ============================================================================
# 5. PREPARE X, y, X_test
# ============================================================================
feature_cols = [c for c in train.columns if c not in ['Id', 'SalePrice']]
X = train[feature_cols].fillna(0)
y = np.log1p(train['SalePrice'])
X_test = test[feature_cols].fillna(0)

print(f"Features count: {len(feature_cols)}")

# Remove extreme outliers (same as best model)
outlier_cond = (X['GrLivArea'] > 4000) & (np.expm1(y) < 300000)
outlier_cond |= (X['TotalBsmtSF'] > 3000) & (np.expm1(y) < 200000)
print(f"Outliers removed: {outlier_cond.sum()}")
X = X[~outlier_cond]
y = y[~outlier_cond]


# ============================================================================
# 6. OPTUNA HYPERPARAMETER OPTIMIZATION (quick, 20 trials each)
# ============================================================================
def optimize_model(name, X_train, y_train, n_trials=20):
    def objective(trial):
        if name == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 600),
                'max_depth': trial.suggest_int('max_depth', 10, 25),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            }
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        elif name == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 3.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),
            }
            model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        elif name == 'lgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 40),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 3.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),
            }
            model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        elif name == 'gb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            }
            model = GradientBoostingRegressor(**params, random_state=42)
        elif name == 'ridge':
            params = {'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True)}
            model = Ridge(**params, random_state=42)
        elif name == 'et':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 600),
                'max_depth': trial.suggest_int('max_depth', 10, 25),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            }
            model = ExtraTreesRegressor(**params, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model {name}")
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())
        return -rmse

    study = optuna.create_study(direction='maximize', study_name=f'{name}_opt')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, -study.best_value


import xgboost as xgb
import lightgbm as lgb

best_params = {}
for m in ['rf', 'xgb', 'lgb', 'gb', 'ridge', 'et']:
    print(f"\n🔍 Optimizing {m.upper()}...")
    params, score = optimize_model(m, X.values, y.values, n_trials=20)
    best_params[m] = params
    print(f"Best {m} RMSLE: {score:.5f}")

# ============================================================================
# 7. TRAIN BASE MODELS ON FULL DATA (using best params)
# ============================================================================
rf = RandomForestRegressor(**best_params['rf'], random_state=42, n_jobs=-1)
xgb_m = xgb.XGBRegressor(**best_params['xgb'], random_state=42, verbosity=0)
lgb_m = lgb.LGBMRegressor(**best_params['lgb'], random_state=42, verbose=-1)
gb = GradientBoostingRegressor(**best_params['gb'], random_state=42)
ridge = Ridge(**best_params['ridge'], random_state=42)
et = ExtraTreesRegressor(**best_params['et'], random_state=42, n_jobs=-1)

base_models = [
    ('rf', rf), ('xgb', xgb_m), ('lgb', lgb_m),
    ('gb', gb), ('ridge', ridge), ('et', et)
]

# ============================================================================
# 8. WEIGHT OPTIMIZATION ON VALIDATION SET (20% hold-out)
# ============================================================================
print("\n📊 Optimizing weights on validation set (20% hold-out)...")
X_train_w, X_val, y_train_w, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train each base model on X_train_w
val_preds = np.zeros((len(X_val), len(base_models)))
for i, (name, model) in enumerate(base_models):
    model.fit(X_train_w, y_train_w)
    val_preds[:, i] = model.predict(X_val)


def rmsle_objective(weights, preds, y_true):
    weighted = np.dot(preds, weights)
    weighted = np.maximum(weighted, 0)  # avoid negative predictions
    return np.sqrt(mean_squared_log_error(y_true, weighted))


# initial equal weights
init_weights = np.ones(len(base_models)) / len(base_models)
bounds = [(0, 1) for _ in range(len(base_models))]
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

result = minimize(
    rmsle_objective, init_weights, args=(val_preds, y_val),
    method='SLSQP', bounds=bounds, constraints=constraints
)
optimal_weights = result.x

print("\nOptimal weights:")
for (name, _), w in zip(base_models, optimal_weights):
    print(f"   {name}: {w:.4f}")
print(f"Validation RMSLE with optimal weights: {result.fun:.5f}")

# ============================================================================
# 9. RETRAIN MODELS ON FULL TRAINING DATA AND MAKE PREDICTIONS
# ============================================================================
print("\n🔄 Retraining models on full training data...")
full_preds_test = np.zeros((len(X_test), len(base_models)))
for i, (name, model) in enumerate(base_models):
    model.fit(X, y)
    full_preds_test[:, i] = model.predict(X_test)

final_log = np.dot(full_preds_test, optimal_weights)
final_log = np.maximum(final_log, 0)
final_pred = np.expm1(final_log)

# ============================================================================
# 10. SAVE SUBMISSION AND MLflow LOGGING
# ============================================================================
sub = pd.DataFrame({'Id': test['Id'], 'SalePrice': final_pred})
sub_filename = 'submission_weighted_ensemble.csv'
sub.to_csv(sub_filename, index=False)
print(f"\n✅ Submission saved: {sub_filename}")
print(f"Statistics: min=${final_pred.min():,.0f}, max=${final_pred.max():,.0f}, mean=${final_pred.mean():,.0f}")

# MLflow logging
try:
    mlflow.set_experiment("house_prices_weighted_ensemble")
    with mlflow.start_run(run_name="weighted_ensemble_6models"):
        # Log optimal weights
        for (name, _), w in zip(base_models, optimal_weights):
            mlflow.log_param(f"weight_{name}", w)
        # Log best hyperparameters
        for model, params in best_params.items():
            for k, v in params.items():
                mlflow.log_param(f"{model}_{k}", v)
        mlflow.log_metric("val_rmsle", result.fun)
        mlflow.log_artifact(sub_filename)
    print("MLflow run completed.")
except Exception as e:
    print(f"MLflow skipped: {e}")

print("\n" + "=" * 80)
print("🚀 READY FOR KAGGLE UPLOAD – EXPECTED RMSLE: ~0.116-0.118")
print("=" * 80)