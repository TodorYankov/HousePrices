import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# Инсталирайте ако нямате
# pip install xgboost lightgbm

import xgboost as xgb
import lightgbm as lgb

print("=" * 80)
print("HOUSE PRICES - STACKING МОДЕЛ (LightGBM + XGBoost + Random Forest)")
print("=" * 80)

# 1. ЗАРЕЖДАНЕ НА ДАННИТЕ
print("\n1. Зареждане на данни...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"   Train: {train.shape}")
print(f"   Test: {test.shape}")

# 2. ОБРАБОТКА НА ЛИПСВАЩИ СТОЙНОСТИ
print("\n2. Обработка на липсващи стойности...")


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

# 3. РАЗШИРЕНО FEATURE ENGINEERING
print("\n3. Създаване на нови features...")


def add_features(df):
    data = df.copy()

    # Обща площ
    data['TotalSF'] = data['TotalBsmtSF'] + data['GrLivArea']
    data['TotalSF_log'] = np.log1p(data['TotalSF'])

    # Обща баня
    data['TotalBath'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']

    # Общо качество
    if 'OverallQual' in data.columns and 'OverallCond' in data.columns:
        data['OverallScore'] = data['OverallQual'] * data['OverallCond']
        data['OverallQual_sq'] = data['OverallQual'] ** 2

    # Възраст
    if 'YearBuilt' in data.columns:
        data['HouseAge'] = 2025 - data['YearBuilt']
        data['HouseAge_sq'] = data['HouseAge'] ** 2

    if 'YearRemodAdd' in data.columns:
        data['YearsSinceRemod'] = 2025 - data['YearRemodAdd']

    # Брой веранди
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    data['TotalPorchSF'] = sum([data[col].fillna(0) for col in porch_cols if col in data.columns])

    # Екстри
    data['HasBsmt'] = (data['TotalBsmtSF'] > 0).astype(int)
    data['HasGarage'] = (data['GarageArea'] > 0).astype(int)
    data['HasFireplace'] = (data['Fireplaces'] > 0).astype(int)
    data['HasPool'] = (data['PoolArea'] > 0).astype(int)
    data['HasDeck'] = (data['WoodDeckSF'] > 0).astype(int)

    # Интеракции
    data['Qual_Area'] = data['OverallQual'] * data['GrLivArea']
    data['Qual_TotalSF'] = data['OverallQual'] * data['TotalSF']
    data['Age_Qual'] = data['HouseAge'] * data['OverallQual']

    # Топ квартали
    if 'Neighborhood' in data.columns:
        top_neighborhoods = ['StoneBr', 'NridgHt', 'NoRidge']
        data['TopNeighborhood'] = data['Neighborhood'].isin(top_neighborhoods).astype(int)

    # Сезонност
    if 'MoSold' in data.columns:
        data['Spring'] = data['MoSold'].isin([3, 4, 5]).astype(int)
        data['Summer'] = data['MoSold'].isin([6, 7, 8]).astype(int)
        data['Fall'] = data['MoSold'].isin([9, 10, 11]).astype(int)

    return data


train = add_features(train)
test = add_features(test)

# 4. КОДИРАНЕ НА КАТЕГОРИЙНИТЕ FEATURES
print("\n4. Кодиране на категорийни features...")

cat_cols = train.select_dtypes(include=['object']).columns
print(f"   Категорийни features: {len(cat_cols)}")

for col in cat_cols:
    if col in train.columns and col != 'SalePrice':
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

# 5. ПОДГОТОВКА НА ДАННИТЕ
print("\n5. Подготовка на данните...")

feature_cols = [col for col in train.columns if col not in ['Id', 'SalePrice']]

X_train = train[feature_cols].fillna(0)
y_train = np.log1p(train['SalePrice'])
X_test = test[feature_cols].fillna(0)

print(f"   Брой features: {len(feature_cols)}")

# 6. STACKING МОДЕЛ
print("\n6. Създаване на Stacking модел...")

# Параметри за всеки модел
rf_params = {'n_estimators': 200, 'max_depth': 20, 'random_state': 42}
xgb_params = {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6, 'random_state': 42}
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

# Кръстосана валидация за stacking
print("\n   Кръстосана валидация (5-fold)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Масиви за съхранение на прогнозите
oof_rf = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
test_preds_rf = np.zeros((len(X_test), 5))
test_preds_xgb = np.zeros((len(X_test), 5))
test_preds_lgb = np.zeros((len(X_test), 5))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"   Fold {fold + 1}/5...")

    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    # Random Forest
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_tr, y_tr)
    oof_rf[val_idx] = rf.predict(X_val)
    test_preds_rf[:, fold] = rf.predict(X_test)

    # XGBoost
    xg = xgb.XGBRegressor(**xgb_params)
    xg.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xg.predict(X_val)
    test_preds_xgb[:, fold] = xg.predict(X_test)

    # LightGBM
    lg = lgb.LGBMRegressor(**lgb_params, n_estimators=1000)
    lg.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lg.predict(X_val)
    test_preds_lgb[:, fold] = lg.predict(X_test)

# Средни прогнози за тест
test_preds_rf_mean = test_preds_rf.mean(axis=1)
test_preds_xgb_mean = test_preds_xgb.mean(axis=1)
test_preds_lgb_mean = test_preds_lgb.mean(axis=1)

# 7. ВТОРИ ЕТАП (Meta Model)
print("\n7. Обучение на Meta модел (Linear Regression)...")

from sklearn.linear_model import LinearRegression

# Подготовка на meta features
meta_train = pd.DataFrame({
    'rf': oof_rf,
    'xgb': oof_xgb,
    'lgb': oof_lgb
})

meta_test = pd.DataFrame({
    'rf': test_preds_rf_mean,
    'xgb': test_preds_xgb_mean,
    'lgb': test_preds_lgb_mean
})

# Meta модел
meta_model = LinearRegression()
meta_model.fit(meta_train, y_train)

# Прогнози
final_predictions = meta_model.predict(meta_test)

# 8. СЪЗДАВАНЕ НА CSV
print("\n8. Създаване на CSV файл...")
predictions = np.expm1(final_predictions)

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': predictions
})
submission.to_csv('submission_house_prices_stacking.csv', index=False)

print(f"\n✅ Файл: submission_house_prices_stacking.csv")
print(f"\n📊 СТАТИСТИКА НА ПРОГНОЗИТЕ:")
print(f"   Минимална: ${predictions.min():,.0f}")
print(f"   Максимална: ${predictions.max():,.0f}")
print(f"   Средна: ${predictions.mean():,.0f}")
print(f"   Медианна: ${np.median(predictions):,.0f}")

# 9. СРАВНЕНИЕ
print("\n" + "=" * 80)
print("📈 ФИНАЛНО СРАВНЕНИЕ НА ВСИЧКИ МОДЕЛИ:")
print("=" * 80)
print(f"   1. Random Forest (11 features)     : 0.15793")
print(f"   2. XGBoost (70+ features)          : 0.12854")
print(f"   3. LightGBM (80+ features)         : 0.12505")
print(f"   4. STACKING (RF + XGB + LGB)       : ОЧАКВА СЕ 0.122 - 0.125")
print("=" * 80)

print("\n💡 ПРОГНОЗА ЗА KAGGLE:")
print("   🎉 ОЧАКВАМ 0.122 - 0.124!")
print("   Stacking често дава най-добър резултат!")

print("\n🚀 ГОТОВО ЗА КАЧВАНЕ В KAGGLE!")
print("📁 Файл: submission_house_prices_stacking.csv")
