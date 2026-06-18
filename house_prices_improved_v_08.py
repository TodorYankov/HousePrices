# house_prices_improved_v_08.py
# Финален модел с 7 базови модела + Ridge(alpha=0.5) мета-модел
# Базиран на резултатите от test_meta_models_with_validation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

print("=" * 80)
print("🏠 HOUSE PRICES - ФИНАЛЕН МОДЕЛ V_08")
print("   7 базови модела + Ridge(alpha=0.5) мета-модел")
print("   Цел: RMSLE < 0.11974")
print("=" * 80)

# ============================================================================
# 1. ЗАРЕЖДАНЕ НА ДАННИ
# ============================================================================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print(f"\nTrain: {train.shape}, Test: {test.shape}")

# ============================================================================
# 2. ОБРАБОТКА НА ЛИПСВАЩИ СТОЙНОСТИ
# ============================================================================
def fix_missing_values(df):
    data = df.copy()
    if 'LotFrontage' in data.columns and 'Neighborhood' in data.columns:
        data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))
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

train = fix_missing_values(train)
test = fix_missing_values(test)

# ============================================================================
# 3. FEATURE ENGINEERING (същите като в най-добрия модел)
# ============================================================================
def add_features(df, neigh_median=None):
    data = df.copy()
    data['TotalSF'] = data['TotalBsmtSF'] + data['GrLivArea']
    data['TotalSF_log'] = np.log1p(data['TotalSF'])
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
    data['AreaPerRoom'] = data['GrLivArea'] / (data['TotRmsAbvGrd'] + 1)
    data['BsmtRatio'] = data['TotalBsmtSF'] / (data['GrLivArea'] + 1)
    data['GarageRatio'] = data['GarageArea'] / (data['GrLivArea'] + 1)
    qual_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0}
    if 'KitchenQual' in data.columns:
        data['KitchenQual_num'] = data['KitchenQual'].map(qual_map).fillna(0)
    if 'BsmtQual' in data.columns:
        data['BsmtQual_num'] = data['BsmtQual'].map(qual_map).fillna(0)
    if 'ExterQual' in data.columns:
        data['ExterQual_num'] = data['ExterQual'].map(qual_map).fillna(0)
    if 'FireplaceQu' in data.columns:
        data['FireplaceQu_num'] = data['FireplaceQu'].map(qual_map).fillna(0)
    data['Qual_Area'] = data['OverallQual'] * data['GrLivArea']
    data['Qual_TotalSF'] = data['OverallQual'] * data['TotalSF']
    data['Age_Qual'] = data['HouseAge'] * data['OverallQual']
    data['Qual_PerAge'] = data['OverallQual'] / (data['HouseAge'] + 1)
    if 'SalePrice' in data.columns:
        neigh_median = data.groupby('Neighborhood')['SalePrice'].median().to_dict()
        data['NeighborhoodMedian'] = data['Neighborhood'].map(neigh_median)
    elif neigh_median is not None:
        data['NeighborhoodMedian'] = data['Neighborhood'].map(neigh_median)
        data['NeighborhoodMedian'] = data['NeighborhoodMedian'].fillna(180000)
    top_neigh = ['StoneBr', 'NridgHt', 'NoRidge', 'Somerst']
    data['TopNeighborhood'] = data['Neighborhood'].isin(top_neigh).astype(int)
    if 'MoSold' in data.columns:
        data['Spring'] = data['MoSold'].isin([3,4,5]).astype(int)
        data['Summer'] = data['MoSold'].isin([6,7,8]).astype(int)
        data['Fall'] = data['MoSold'].isin([9,10,11]).astype(int)
        data['Winter'] = data['MoSold'].isin([12,1,2]).astype(int)
    expensive = ['BrkFace', 'Stone', 'BrkComm', 'CBlock']
    if 'Exterior1st' in data.columns:
        data['ExpensiveMaterial'] = data['Exterior1st'].isin(expensive).astype(int)
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    data['TotalPorchSF'] = sum([data[col].fillna(0) for col in porch_cols if col in data.columns])
    data['HasPorch'] = (data['TotalPorchSF'] > 0).astype(int)
    return data, neigh_median

train, neigh_median_dict = add_features(train)
test, _ = add_features(test, neigh_median=neigh_median_dict)

# ============================================================================
# 4. КОДИРАНЕ НА КАТЕГОРИЙНИТЕ
# ============================================================================
cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col != 'SalePrice':
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

# ============================================================================
# 5. ПОДГОТОВКА НА X, y
# ============================================================================
feature_cols = [c for c in train.columns if c not in ['Id', 'SalePrice']]
X = train[feature_cols].fillna(0)
y = np.log1p(train['SalePrice'])

# Премахване на аутлайъри
outlier_cond = (X['GrLivArea'] > 4000) & (np.expm1(y) < 300000)
outlier_cond |= (X['TotalBsmtSF'] > 3000) & (np.expm1(y) < 200000)
X = X[~outlier_cond]
y = y[~outlier_cond]
print(f"Премахнати аутлайъри: {outlier_cond.sum()}")

# Скалиране
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test = test[feature_cols].fillna(0)
X_test_scaled = scaler.transform(X_test)

print(f"Брой features: {len(feature_cols)}")

# ============================================================================
# 6. ОБУЧЕНИЕ НА БАЗОВИТЕ МОДЕЛИ (7 модела)
# ============================================================================
print("\n📊 Обучение на 7 базови модела с 5-fold CV...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Параметри (същите като в test_meta_models_enhanced.py)
rf_params = {'n_estimators': 500, 'max_depth': 25, 'min_samples_split': 5,
             'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1}
xgb_params = {'n_estimators': 1000, 'learning_rate': 0.02, 'max_depth': 6,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
              'verbosity': 0}
lgb_params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
              'num_leaves': 63, 'learning_rate': 0.02, 'feature_fraction': 0.8,
              'bagging_fraction': 0.8, 'bagging_freq': 5, 'n_estimators': 1500,
              'verbose': -1, 'random_state': 42}
gb_params = {'n_estimators': 500, 'learning_rate': 0.02, 'max_depth': 5,
             'subsample': 0.8, 'random_state': 42}
ridge_params = {'alpha': 1.0, 'random_state': 42}
cat_params = {'iterations': 500, 'learning_rate': 0.03, 'depth': 6,
              'l2_leaf_reg': 3, 'random_seed': 42, 'verbose': False}
et_params = {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 5,
             'random_state': 42, 'n_jobs': -1}

models = [
    ('rf', RandomForestRegressor(**rf_params)),
    ('xgb', xgb.XGBRegressor(**xgb_params)),
    ('lgb', lgb.LGBMRegressor(**lgb_params)),
    ('gb', GradientBoostingRegressor(**gb_params)),
    ('ridge', Ridge(**ridge_params)),
    ('cat', CatBoostRegressor(**cat_params)),
    ('et', ExtraTreesRegressor(**et_params))
]

# OOF прогнози и тестови прогнози
oof_preds = np.zeros((len(X_scaled), len(models)))
test_preds = np.zeros((len(X_test_scaled), len(models), 5))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"   Fold {fold+1}/5")
    X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    for i, (name, model) in enumerate(models):
        model.fit(X_tr, y_tr)
        oof_preds[val_idx, i] = model.predict(X_val)
        test_preds[:, i, fold] = model.predict(X_test_scaled)

# Средни тестови прогнози
test_preds_mean = test_preds.mean(axis=2)  # (n_test, n_models)

# ============================================================================
# 7. МЕТА-МОДЕЛ: Ridge(alpha=0.5)
# ============================================================================
print("\n🔧 Обучение на мета-модел: Ridge(alpha=0.5)...")
meta_model = Ridge(alpha=0.5, random_state=42)
meta_model.fit(oof_preds, y)

# Коефициенти
print("\nКоефициенти на мета-модела:")
for name, coef in zip([name for name, _ in models], meta_model.coef_):
    print(f"   {name}: {coef:.4f}")

# Прогнози
final_log = meta_model.predict(test_preds_mean)
final_predictions = np.expm1(final_log)

# ============================================================================
# 8. SUBMISSION
# ============================================================================
print("\n📁 Създаване на submission файл...")
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': final_predictions
})
submission.to_csv('submission_house_prices_improved_v_08.csv', index=False)

print("\n" + "=" * 80)
print("✅ ФИНАЛЕН МОДЕЛ V_08 ЗАВЪРШЕН!")
print("=" * 80)
print(f"📊 Статистика на прогнозите:")
print(f"   Минимална: ${final_predictions.min():,.0f}")
print(f"   Максимална: ${final_predictions.max():,.0f}")
print(f"   Средна: ${final_predictions.mean():,.0f}")
print(f"   Медианна: ${np.median(final_predictions):,.0f}")
print(f"\n📈 Очакван RMSLE: под 0.11974 (възможно ~0.1196)")
print(f"📁 Файл: submission_house_prices_improved_v_08.csv")
print("🚀 Качете файла в Kaggle за резултат!")
print("=" * 80)

