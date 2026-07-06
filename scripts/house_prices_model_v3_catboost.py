# house_prices_model_v3_catboost.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("HOUSE PRICES MODEL V3 - CATBOOST + FEATURE SELECTION")
print("=" * 80)

# 1. ЗАРЕЖДАНЕ
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")


# 2. ОСНОВНА ПОДГОТОВКА (от вашия improved модел)
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


def add_features(df):
    data = df.copy()
    data['TotalSF'] = data['TotalBsmtSF'] + data['GrLivArea']
    data['TotalBath'] = (data['FullBath'] + 0.5 * data['HalfBath'] +
                         data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath'])
    if 'YearBuilt' in data.columns:
        data['HouseAge'] = 2010 - data['YearBuilt']
    data['Qual_Area'] = data['OverallQual'] * data['GrLivArea']
    data['Qual_TotalSF'] = data['OverallQual'] * data['TotalSF']
    data['HasBsmt'] = (data['TotalBsmtSF'] > 0).astype(int)
    data['HasGarage'] = (data['GarageArea'] > 0).astype(int)
    data['TopNeighborhood'] = data['Neighborhood'].isin(['StoneBr', 'NridgHt', 'NoRidge']).astype(int)
    return data


# Прилагаме
train = fix_missing_values(train)
test = fix_missing_values(test)
train = add_features(train)
test = add_features(test)

# Кодиране на категорийни
cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Подготовка
feature_cols = [col for col in train.columns if col not in ['Id', 'SalePrice']]
X_full = train[feature_cols].fillna(0)
y = np.log1p(train['SalePrice'])
X_test = test[feature_cols].fillna(0)

print(f"Първоначален брой features: {X_full.shape[1]}")

# 3. FEATURE SELECTION с Lasso
print("\nFeature selection с Lasso...")
lasso = Lasso(alpha=0.0005, max_iter=5000, random_state=42)
lasso.fit(X_full, y)
selected = lasso.coef_ != 0
selected_features = X_full.columns[selected]
X = X_full[selected_features]
X_test = X_test[selected_features]

print(f"Избрани features: {len(selected_features)} (от {X_full.shape[1]})")

# 4. OUTLIER REMOVAL
print("\nПремахване на outliers...")
outliers = (X['GrLivArea'] > 4000) & (np.expm1(y) < 300000)
outliers = outliers | ((X['TotalBsmtSF'] > 3000) & (np.expm1(y) < 200000))
print(f"Премахнати: {outliers.sum()} outliers")
X = X[~outliers]
y = y[~outliers]

# 5. НОРМАЛИЗАЦИЯ
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 6. CATBOOST + STACKING
print("\nОбучение на ансамбъл с CatBoost...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Модели
rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, random_state=42, verbosity=0)
lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
cat = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=5, random_seed=42, verbose=False)

models = [('rf', rf), ('xgb', xgb_model), ('lgb', lgb_model), ('cat', cat)]

# OOF прогнози
oof_preds = {name: np.zeros(len(X)) for name, _ in models}
test_preds = {name: np.zeros((len(X_test), 5)) for name, _ in models}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"   Fold {fold + 1}/5")
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    for name, model in models:
        model.fit(X_tr, y_tr)
        oof_preds[name][val_idx] = model.predict(X_val)
        test_preds[name][:, fold] = model.predict(X_test_scaled)

# Мета features
meta_train = pd.DataFrame({name: oof_preds[name] for name, _ in models})
meta_test = pd.DataFrame({name: test_preds[name].mean(axis=1) for name, _ in models})

# Мета модел (Ridge)
meta_model = Ridge(alpha=0.5, random_state=42)
meta_model.fit(meta_train, y)

# Първична прогноза
primary_pred = meta_model.predict(meta_test)

# 7. 2-STAGE: RF на residuals
print("\n2-stage prediction (RF на residuals)...")
train_pred = meta_model.predict(meta_train)
residuals = y - train_pred

rf_residuals = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
rf_residuals.fit(meta_train, residuals)
residuals_pred = rf_residuals.predict(meta_test)

# Финален prediction
final_predictions_log = primary_pred + residuals_pred
final_predictions = np.expm1(final_predictions_log)

# 8. ЗАПИСВАНЕ
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': final_predictions})
submission.to_csv('house_prices_model_v3_catboost.csv', index=False)

print("\n" + "=" * 80)
print("📊 РЕЗУЛТАТИ")
print("=" * 80)
print(f"   Брой features: {len(selected_features)}")
print(f"   Модели в ансамбъла: RF, XGB, LGB, CatBoost")
print(f"   Мета модел: Ridge")
print(f"   2-stage: RF на residuals")
print(f"\n📊 Статистика на прогнозите:")
print(f"   Минимална: ${final_predictions.min():,.0f}")
print(f"   Максимална: ${final_predictions.max():,.0f}")
print(f"   Средна: ${final_predictions.mean():,.0f}")

print("\n🚀 ГОТОВО ЗА КАЧВАНЕ!")
print("📁 Файл: house_prices_model_v3_catboost.csv")
print("=" * 80)
