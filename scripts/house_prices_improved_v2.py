# house_prices_improved_v2.py
# Final improved stacking model – target RMSLE < 0.115
# Based on your best model (0.11988) – adds Optuna, advanced features, Lasso meta, residual correction

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import optuna
import mlflow

print("=" * 80)
print("🏆 FINAL HOUSE PRICES MODEL – TARGET RMSLE < 0.115")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================================
# 2. MISSING VALUES (improved)
# ============================================================================
def fix_missing(df, is_train=True, neigh_median_dict=None):
    data = df.copy()
    if 'LotFrontage' in data.columns and 'Neighborhood' in data.columns:
        if is_train:
            data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median()))
        else:
            if neigh_median_dict is not None:
                data['LotFrontage'] = data.apply(
                    lambda row: neigh_median_dict.get(row['Neighborhood'], row['LotFrontage'])
                    if pd.isna(row['LotFrontage']) else row['LotFrontage'], axis=1)
            else:
                data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
    num_features = ['MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2',
                    'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                    'GarageArea', 'GarageCars']
    for col in num_features:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    if 'GarageYrBlt' in data.columns and 'YearBuilt' in data.columns:
        data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['YearBuilt'])
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

train = fix_missing(train, is_train=True)
test = fix_missing(test, is_train=False)

# ============================================================================
# 3. ADVANCED FEATURE ENGINEERING
# ============================================================================
def add_features(df, is_train=True, neigh_median=None):
    data = df.copy()
    # Basic features from your best model
    data['TotalSF'] = data['TotalBsmtSF'] + data['GrLivArea']
    data['TotalBath'] = data['FullBath'] + 0.5*data['HalfBath'] + data['BsmtFullBath'] + 0.5*data['BsmtHalfBath']
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
    # Quality numerical
    qual_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0}
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
    # Neighborhood median
    if is_train and 'Neighborhood' in data.columns and 'SalePrice' in data.columns:
        neigh_median = data.groupby('Neighborhood')['SalePrice'].median().to_dict()
        data['NeighborhoodMedian'] = data['Neighborhood'].map(neigh_median)
    elif not is_train and neigh_median is not None:
        data['NeighborhoodMedian'] = data['Neighborhood'].map(neigh_median)
        data['NeighborhoodMedian'] = data['NeighborhoodMedian'].fillna(180000)
    # Top neighborhoods
    top_neigh = ['StoneBr', 'NridgHt', 'NoRidge', 'Somerst']
    data['TopNeighborhood'] = data['Neighborhood'].isin(top_neigh).astype(int)
    # Seasonality
    if 'MoSold' in data.columns:
        data['Spring'] = data['MoSold'].isin([3,4,5]).astype(int)
        data['Summer'] = data['MoSold'].isin([6,7,8]).astype(int)
        data['Fall'] = data['MoSold'].isin([9,10,11]).astype(int)
        data['Winter'] = data['MoSold'].isin([12,1,2]).astype(int)
    # Expensive material
    expensive = ['BrkFace', 'Stone', 'BrkComm', 'CBlock']
    if 'Exterior1st' in data.columns:
        data['ExpensiveMaterial'] = data['Exterior1st'].isin(expensive).astype(int)
    # Total porch
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    data['TotalPorchSF'] = sum([data[col].fillna(0) for col in porch_cols if col in data.columns])
    data['HasPorch'] = (data['TotalPorchSF'] > 0).astype(int)
    return data, neigh_median if is_train else None

train, neigh_median_dict = add_features(train, is_train=True)
test, _ = add_features(test, is_train=False, neigh_median=neigh_median_dict)

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Remove extreme outliers (same as your best model)
outlier_cond = (X['GrLivArea'] > 4000) & (np.expm1(y) < 300000)
outlier_cond |= (X['TotalBsmtSF'] > 3000) & (np.expm1(y) < 200000)
print(f"Outliers removed: {outlier_cond.sum()}")
X_scaled = X_scaled[~outlier_cond]
y = y[~outlier_cond]

# ============================================================================
# 6. OPTUNA HYPERPARAMETER OPTIMIZATION (50 trials each)
# ============================================================================
def optimize_model(name, X_train, y_train, n_trials=50):
    def objective(trial):
        if name == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'max_depth': trial.suggest_int('max_depth', 10, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            }
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        elif name == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            }
            model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        elif name == 'lgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            }
            model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        elif name == 'cat':
            params = {
                'iterations': trial.suggest_int('iterations', 500, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            }
            model = CatBoostRegressor(**params, random_seed=42, verbose=False)
        elif name == 'ridge':
            params = {'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True)}
            model = RidgeCV(alphas=[params['alpha']], cv=3)
        else:
            raise ValueError(f"Unknown model {name}")
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())
        return -rmse
    study = optuna.create_study(direction='maximize', study_name=f'{name}_opt')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, -study.best_value

best_params = {}
for m in ['rf', 'xgb', 'lgb', 'cat', 'ridge']:
    print(f"\n🔍 Optimizing {m.upper()}...")
    params, score = optimize_model(m, X_scaled, y, n_trials=50)
    best_params[m] = params
    print(f"Best {m} RMSLE: {score:.5f}")

# ============================================================================
# 7. STACKING WITH 5 MODELS + Lasso META-LEARNER
# ============================================================================
print("\n📊 Training stacking ensemble (5 models + Lasso meta)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestRegressor(**best_params['rf'], random_state=42, n_jobs=-1)
xgb_m = xgb.XGBRegressor(**best_params['xgb'], random_state=42, verbosity=0)
lgb_m = lgb.LGBMRegressor(**best_params['lgb'], random_state=42, verbose=-1)
cat_m = CatBoostRegressor(**best_params['cat'], random_seed=42, verbose=False)
ridge_m = RidgeCV(alphas=[best_params['ridge']['alpha']], cv=3)

models = [('rf', rf), ('xgb', xgb_m), ('lgb', lgb_m), ('cat', cat_m), ('ridge', ridge_m)]

oof = {name: np.zeros(len(X_scaled)) for name, _ in models}
test_preds = {name: np.zeros((len(X_test_scaled), 5)) for name, _ in models}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"   Fold {fold+1}/5")
    X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    for name, model in models:
        model.fit(X_tr, y_tr)
        oof[name][val_idx] = model.predict(X_val)
        test_preds[name][:, fold] = model.predict(X_test_scaled)

meta_train = pd.DataFrame({name: oof[name] for name, _ in models})
meta_test = pd.DataFrame({name: test_preds[name].mean(axis=1) for name, _ in models})

# Lasso meta-learner (automatically selects best models)
meta_model = LassoCV(cv=5, random_state=42, max_iter=5000)
meta_model.fit(meta_train, y)
primary = meta_model.predict(meta_test)

print("\nMeta-learner coefficients (Lasso):")
for name, coef in zip([name for name, _ in models], meta_model.coef_):
    print(f"   {name}: {coef:.5f}")
nonzero = np.sum(meta_model.coef_ != 0)
print(f"Non-zero coefficients: {nonzero} of {len(models)}")

# ============================================================================
# 8. RESIDUAL CORRECTION (LGBM on residuals)
# ============================================================================
print("\n🔄 Second‑stage: correcting residuals with LightGBM...")
train_pred = meta_model.predict(meta_train)
residuals = y - train_pred

res_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
res_model.fit(meta_train, residuals)
residuals_pred = res_model.predict(meta_test)

final_log = primary + residuals_pred
final_pred = np.expm1(final_log)

# ============================================================================
# 9. SUBMISSION AND MLflow LOGGING
# ============================================================================
sub = pd.DataFrame({'Id': test['Id'], 'SalePrice': final_pred})
sub_filename = 'house_prices_improved_v2.csv'
sub.to_csv(sub_filename, index=False)
print(f"\n✅ Submission saved: {sub_filename}")
print(f"Statistics: min=${final_pred.min():,.0f}, max=${final_pred.max():,.0f}, mean=${final_pred.mean():,.0f}")

# Optional MLflow logging (skip if file not found)
try:
    mlflow.set_experiment("house_prices_final")
    with mlflow.start_run(run_name="stacking_5models_lasso_residuals"):
        # Log parameters
        for model, params in best_params.items():
            for k, v in params.items():
                mlflow.log_param(f"{model}_{k}", v)
        # Log meta-learner coefficients
        for name, coef in zip([name for name, _ in models], meta_model.coef_):
            mlflow.log_metric(f"coef_{name}", coef)
        mlflow.log_metric("train_rmse", np.sqrt(mean_squared_error(y, train_pred)))
        mlflow.log_metric("final_rmse", np.sqrt(mean_squared_error(y, train_pred + res_model.predict(meta_train))))
        if os.path.exists(sub_filename):
            mlflow.log_artifact(sub_filename)
        else:
            print("Submission file not found for MLflow artifact.")
    print("MLflow run completed.")
except Exception as e:
    print(f"MLflow logging skipped: {e}")

print("\n" + "=" * 80)
print("🚀 READY FOR KAGGLE UPLOAD")
print("=" * 80)