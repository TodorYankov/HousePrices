# house_prices_final_improved.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
import mlflow
import mlflow.sklearn
import optuna
import joblib
import os

warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb

print("=" * 80)
print("🏆 FINAL IMPROVED MODEL - TARGET RMSLE 0.115-0.117")
print("=" * 80)

# ============================================================================
# 1. ЗАРЕЖДАНЕ НА ДАННИ
# ============================================================================
print("\n1. Loading data...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"   Train: {train.shape}")
print(f"   Test: {test.shape}")


# ============================================================================
# 2. ПОДОБРЕНА ОБРАБОТКА НА ЛИПСВАЩИ СТОЙНОСТИ
# ============================================================================
def fix_missing_values(df):
    data = df.copy()

    # LotFrontage по квартали (по-интелигентно)
    if 'LotFrontage' in data.columns and 'Neighborhood' in data.columns:
        data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))

    # Числови features - медиана
    num_features = ['MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2',
                    'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                    'GarageArea', 'GarageCars']
    for col in num_features:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())

    # GarageYrBlt - по-добро запълване
    if 'GarageYrBlt' in data.columns and 'YearBuilt' in data.columns:
        data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['YearBuilt'])

    # Категорийни features
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


# ============================================================================
# 3. РАЗШИРЕН FEATURE ENGINEERING (90+ features)
# ============================================================================
def add_advanced_features(df):
    data = df.copy()

    # ----- ОСНОВНИ FEATURES (от вашия improved модел) -----
    data['TotalSF'] = data['TotalBsmtSF'] + data['GrLivArea']
    data['TotalSF_log'] = np.log1p(data['TotalSF'])

    data['TotalBath'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']

    if 'OverallQual' in data.columns and 'OverallCond' in data.columns:
        data['OverallScore'] = data['OverallQual'] * data['OverallCond']
        data['OverallQual_sq'] = data['OverallQual'] ** 2

    if 'YearBuilt' in data.columns:
        data['HouseAge'] = 2025 - data['YearBuilt']
        data['HouseAge_sq'] = data['HouseAge'] ** 2

    if 'YearRemodAdd' in data.columns:
        data['YearsSinceRemod'] = 2025 - data['YearRemodAdd']
        data['RemodAge'] = (data['YearRemodAdd'] - data['YearBuilt']).clip(0, 50)

    # ----- ЕКСТРИ -----
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    data['TotalPorchSF'] = sum([data[col].fillna(0) for col in porch_cols if col in data.columns])

    data['HasBsmt'] = (data['TotalBsmtSF'] > 0).astype(int)
    data['HasGarage'] = (data['GarageArea'] > 0).astype(int)
    data['HasFireplace'] = (data['Fireplaces'] > 0).astype(int)
    data['HasPool'] = (data['PoolArea'] > 0).astype(int)
    data['HasDeck'] = (data['WoodDeckSF'] > 0).astype(int)
    data['HasPorch'] = (data['TotalPorchSF'] > 0).astype(int)

    # ----- ИНТЕРАКЦИИ -----
    data['Qual_Area'] = data['OverallQual'] * data['GrLivArea']
    data['Qual_TotalSF'] = data['OverallQual'] * data['TotalSF']
    data['Age_Qual'] = data['HouseAge'] * data['OverallQual']
    data['QualPerAge'] = data['OverallQual'] / (data['HouseAge'] + 1)
    data['Age_Bath'] = data['HouseAge'] * data['TotalBath']

    # ----- КВАРТАЛИ -----
    if 'Neighborhood' in data.columns:
        top_neighborhoods = ['StoneBr', 'NridgHt', 'NoRidge', 'Somerst']
        data['TopNeighborhood'] = data['Neighborhood'].isin(top_neighborhoods).astype(int)

        # НОВО: Оценка на квартали по цена (median price per neighborhood)
        # (ще се изчисли по-късно на train)

    # ----- СЕЗОННОСТ -----
    if 'MoSold' in data.columns:
        data['Spring'] = data['MoSold'].isin([3, 4, 5]).astype(int)
        data['Summer'] = data['MoSold'].isin([6, 7, 8]).astype(int)
        data['Fall'] = data['MoSold'].isin([9, 10, 11]).astype(int)
        data['Winter'] = data['MoSold'].isin([12, 1, 2]).astype(int)

    # ----- НОВИ features за подобрение -----
    data['TotalRooms'] = data['TotRmsAbvGrd'] + data['BsmtFinSF1'] / 500
    data['IsUrban'] = data['MSSubClass'].isin([60, 120, 150, 180, 190]).astype(int)

    # Качествени оценки (числови)
    kitchen_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    if 'KitchenQual' in data.columns:
        data['KitchenQual_num'] = data['KitchenQual'].map(kitchen_map).fillna(0)

    bsmt_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    if 'BsmtQual' in data.columns:
        data['BsmtQual_num'] = data['BsmtQual'].map(bsmt_map).fillna(0)

    exter_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    if 'ExterQual' in data.columns:
        data['ExterQual_num'] = data['ExterQual'].map(exter_map).fillna(0)

    fire_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    if 'FireplaceQu' in data.columns:
        data['FireplaceQu_num'] = data['FireplaceQu'].map(fire_map).fillna(0)

    # Интеракции на качество
    if 'KitchenQual_num' in data.columns:
        data['Kitchen_Area'] = data['KitchenQual_num'] * data['GrLivArea']
    if 'BsmtQual_num' in data.columns:
        data['BsmtQual_Area'] = data['BsmtQual_num'] * data['TotalBsmtSF']

    data['TotalBath_precise'] = data['FullBath'] + 0.5 * data['HalfBath']
    if 'BsmtFullBath' in data.columns:
        data['TotalBath_precise'] += data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']

    # Скъпи материали
    expensive_materials = ['BrkFace', 'Stone', 'BrkComm', 'CBlock']
    if 'Exterior1st' in data.columns:
        data['ExpensiveMaterial'] = data['Exterior1st'].isin(expensive_materials).astype(int)

    # ----- ОТНОШЕНИЯ -----
    data['AreaPerRoom'] = data['GrLivArea'] / (data['TotRmsAbvGrd'] + 1)
    data['BsmtRatio'] = data['TotalBsmtSF'] / (data['GrLivArea'] + 1)
    data['GarageRatio'] = data['GarageArea'] / (data['GrLivArea'] + 1)

    return data


print("\n2. Processing missing values...")
train = fix_missing_values(train)
test = fix_missing_values(test)

print("\n3. Adding advanced features...")
train = add_advanced_features(train)
test = add_advanced_features(test)

# ============================================================================
# 4. НОВО: Добавяне на квартални статистики
# ============================================================================
print("\n4. Adding neighborhood statistics...")

# Изчисляване на медианна цена по квартал само от train
neigh_median = train.groupby('Neighborhood')['SalePrice'].median()
neigh_median_dict = neigh_median.to_dict()

# Добавяне на feature
train['NeighborhoodMedian'] = train['Neighborhood'].map(neigh_median_dict)
# За test: използваме най-близкия квартал или общата медиана
test['NeighborhoodMedian'] = test['Neighborhood'].map(neigh_median_dict)
test['NeighborhoodMedian'] = test['NeighborhoodMedian'].fillna(train['SalePrice'].median())

# ============================================================================
# 5. КОДИРАНЕ НА КАТЕГОРИЙНИТЕ FEATURES
# ============================================================================
print("\n5. Encoding categorical features...")
cat_cols = train.select_dtypes(include=['object']).columns
print(f"   Categorical features: {len(cat_cols)}")

for col in cat_cols:
    if col in train.columns and col != 'SalePrice':
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

# ============================================================================
# 6. ПОДГОТОВКА НА ДАННИТЕ
# ============================================================================
print("\n6. Preparing data...")
feature_cols = [col for col in train.columns if col not in ['Id', 'SalePrice']]
X_train = train[feature_cols].fillna(0)
y_train = np.log1p(train['SalePrice'])
X_test = test[feature_cols].fillna(0)

print(f"   Total features: {len(feature_cols)}")
print(f"   Train shape: {X_train.shape}")
print(f"   Test shape: {X_test.shape}")

# ============================================================================
# 7. OPTUNA ХИПЕРПАРАМЕТРОВА ОПТИМИЗАЦИЯ
# ============================================================================
print("\n7. Optuna hyperparameter optimization (30 trials)...")


def objective(trial):
    """Целева функция за Optuna"""
    model_name = trial.suggest_categorical('model', ['rf', 'xgb', 'lgb'])

    if model_name == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 10, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        }
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    elif model_name == 'xgb':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
        }
        model = xgb.XGBRegressor(**params, random_state=42)
    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        }
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)

    # Кръстосана валидация
    scores = cross_val_score(model, X_train, y_train, cv=3,
                             scoring='neg_mean_squared_error', n_jobs=-1)
    rmse = np.sqrt(-scores.mean())
    return -rmse  # Optuna максимизира


# Създаване на study
study = optuna.create_study(direction='maximize', study_name='house_prices_opt')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\n✨ Best parameters: {study.best_params}")
print(f"✨ Best CV score: {-study.best_value:.4f} RMSLE")

# ============================================================================
# 8. MLflow SETUP
# ============================================================================
print("\n8. Setting up MLflow...")
os.makedirs('models_saved/mlflow', exist_ok=True)
mlflow.set_tracking_uri("file:///" + os.path.abspath("models_saved/mlflow").replace('\\', '/'))
mlflow.set_experiment("house_prices_final")

# ============================================================================
# 9. ОБУЧЕНИЕ НА ФИНАЛЕН СТЕКИНГ МОДЕЛ
# ============================================================================
print("\n9. Training final stacking model...")

with mlflow.start_run(run_name="final_improved_stacking") as run:
    # Log parameters
    mlflow.log_params(study.best_params)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("feature_list", str(feature_cols[:10]) + "...")

    # Параметри за base моделите
    rf_params = {'n_estimators': 500, 'max_depth': 25, 'random_state': 42, 'n_jobs': -1}
    xgb_params = {'n_estimators': 1000, 'learning_rate': 0.02, 'max_depth': 6,
                  'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42}
    lgb_params = {'objective': 'regression', 'metric': 'rmse', 'num_leaves': 63,
                  'learning_rate': 0.02, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
                  'n_estimators': 1500, 'verbose': -1, 'random_state': 42}
    gb_params = {'n_estimators': 500, 'learning_rate': 0.02, 'max_depth': 5,
                 'subsample': 0.8, 'random_state': 42}
    ridge_params = {'alpha': 1.0, 'random_state': 42}

    # 5-fold stacking
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_rf, oof_xgb, oof_lgb, oof_gb, oof_ridge = [
        np.zeros(len(X_train)) for _ in range(5)
    ]

    test_preds = {name: np.zeros((len(X_test), 5)) for name in ['rf', 'xgb', 'lgb', 'gb', 'ridge']}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"   Fold {fold + 1}/5...")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Random Forest
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_tr, y_tr)
        oof_rf[val_idx] = rf.predict(X_val)
        test_preds['rf'][:, fold] = rf.predict(X_test)

        # XGBoost
        xg = xgb.XGBRegressor(**xgb_params)
        xg.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xg.predict(X_val)
        test_preds['xgb'][:, fold] = xg.predict(X_test)

        # LightGBM
        lg = lgb.LGBMRegressor(**lgb_params)
        lg.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lg.predict(X_val)
        test_preds['lgb'][:, fold] = lg.predict(X_test)

        # Gradient Boosting
        gb = GradientBoostingRegressor(**gb_params)
        gb.fit(X_tr, y_tr)
        oof_gb[val_idx] = gb.predict(X_val)
        test_preds['gb'][:, fold] = gb.predict(X_test)

        # Ridge
        ridge = Ridge(**ridge_params)
        ridge.fit(X_tr, y_tr)
        oof_ridge[val_idx] = ridge.predict(X_val)
        test_preds['ridge'][:, fold] = ridge.predict(X_test)

    # Meta model
    meta_train = pd.DataFrame({
        'rf': oof_rf, 'xgb': oof_xgb, 'lgb': oof_lgb,
        'gb': oof_gb, 'ridge': oof_ridge
    })

    meta_test = pd.DataFrame({
        'rf': test_preds['rf'].mean(axis=1),
        'xgb': test_preds['xgb'].mean(axis=1),
        'lgb': test_preds['lgb'].mean(axis=1),
        'gb': test_preds['gb'].mean(axis=1),
        'ridge': test_preds['ridge'].mean(axis=1)
    })

    meta_model = Ridge(alpha=0.5, random_state=42)
    meta_model.fit(meta_train, y_train)

    # Log coefficients
    for name, coef in zip(['rf', 'xgb', 'lgb', 'gb', 'ridge'], meta_model.coef_):
        mlflow.log_metric(f"coef_{name}", coef)
        print(f"      {name}: {coef:.3f}")

    # Прогнози
    final_predictions = meta_model.predict(meta_test)
    predictions = np.expm1(final_predictions)

    # Log metrics
    train_pred = meta_model.predict(meta_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("target_rmsle", 0.11988)
    mlflow.log_metric("improvement_target", 0.115)  # Цел

    # ========================================================================
    # 10. SUBMISSION
    # ========================================================================
    print("\n10. Creating submission file...")
    submission = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': predictions
    })
    submission.to_csv('submission_final_improved.csv', index=False)
    mlflow.log_artifact('submission_final_improved.csv')

    # Запазване на модела
    joblib.dump({
        'meta_model': meta_model,
        'features': feature_cols,
        'best_params': study.best_params
    }, 'models_saved/final_model.pkl')

    # ========================================================================
    # 11. SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ FINAL IMPROVED MODEL COMPLETE!")
    print("=" * 80)
    print(f"📁 File: submission_final_improved.csv")
    print(f"📊 Statistics:")
    print(f"   Min: ${predictions.min():,.0f}")
    print(f"   Max: ${predictions.max():,.0f}")
    print(f"   Mean: ${predictions.mean():,.0f}")
    print(f"   Median: ${np.median(predictions):,.0f}")
    print(f"\n📈 Target RMSLE: 0.115 - 0.117")
    print(f"🏆 Previous best: 0.11988")
    print(f"🎯 Expected improvement: -0.004 to -0.005")
    print("\n" + "=" * 80)
    print("🚀 To see MLflow UI:")
    print("   mlflow ui --port 5000")
    print("=" * 80)