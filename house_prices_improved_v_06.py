# house_prices_improved_v_06.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_log_error
import warnings

warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb

print("=" * 80)
print("HOUSE PRICES - ПОДОБРЕН STACKING МОДЕЛ (5 модела + Meta-Learner)")
print("=" * 80)

# 1. ЗАРЕЖДАНЕ НА ДАННИТЕ
print("\n1. Зареждане на данни...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"   Train: {train.shape}")
print(f"   Test: {test.shape}")

# 2. ОБРАБОТКА НА ЛИПСВАЩИ СТОЙНОСТИ (ПОДОБРЕНА)
print("\n2. Обработка на липсващи стойности...")


def fix_missing_values(df):
    data = df.copy()

    # НОВО: По-интелигентно запълване за LotFrontage (по квартали)
    if 'LotFrontage' in data.columns and 'Neighborhood' in data.columns:
        data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )

    num_features = ['MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1',
                    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
                    'BsmtHalfBath', 'GarageArea', 'GarageCars']

    for col in num_features:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())

    # НОВО: По-добро запълване за GarageYrBlt
    if 'GarageYrBlt' in data.columns:
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

# 3. РАЗШИРЕНО FEATURE ENGINEERING (ДОБАВЕНИ НОВИ FEATURES)
print("\n3. Създаване на нови features...")


def add_features(df):
    data = df.copy()

    # ===== ВАШИТЕ ОРИГИНАЛНИ FEATURES =====
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

    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    data['TotalPorchSF'] = sum([data[col].fillna(0) for col in porch_cols if col in data.columns])

    data['HasBsmt'] = (data['TotalBsmtSF'] > 0).astype(int)
    data['HasGarage'] = (data['GarageArea'] > 0).astype(int)
    data['HasFireplace'] = (data['Fireplaces'] > 0).astype(int)
    data['HasPool'] = (data['PoolArea'] > 0).astype(int)
    data['HasDeck'] = (data['WoodDeckSF'] > 0).astype(int)

    data['Qual_Area'] = data['OverallQual'] * data['GrLivArea']
    data['Qual_TotalSF'] = data['OverallQual'] * data['TotalSF']
    data['Age_Qual'] = data['HouseAge'] * data['OverallQual']

    if 'Neighborhood' in data.columns:
        top_neighborhoods = ['StoneBr', 'NridgHt', 'NoRidge']
        data['TopNeighborhood'] = data['Neighborhood'].isin(top_neighborhoods).astype(int)

    if 'MoSold' in data.columns:
        data['Spring'] = data['MoSold'].isin([3, 4, 5]).astype(int)
        data['Summer'] = data['MoSold'].isin([6, 7, 8]).astype(int)
        data['Fall'] = data['MoSold'].isin([9, 10, 11]).astype(int)

    # ===== НОВИ FEATURES ЗА ПОДОБРЕНИЕ =====

    # 1. Отношение качество/възраст
    data['QualPerAge'] = data['OverallQual'] / (data['HouseAge'] + 1)

    # 2. Общ брой стаи
    data['TotalRooms'] = data['TotRmsAbvGrd'] + data['BsmtFinSF1'] / 500

    # 3. Локация (близост до центъра според MSSubClass)
    data['IsUrban'] = data['MSSubClass'].isin([60, 120, 150, 180, 190]).astype(int)

    # 4. Качество на кухнята като число
    kitchen_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    if 'KitchenQual' in data.columns:
        data['KitchenQual_num'] = data['KitchenQual'].map(kitchen_map).fillna(0)

    # 5. Качество на мазето като число
    bsmt_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    if 'BsmtQual' in data.columns:
        data['BsmtQual_num'] = data['BsmtQual'].map(bsmt_map).fillna(0)

    # 6. Интеракция между качество на кухня и обща площ
    if 'KitchenQual_num' in data.columns:
        data['Kitchen_Area'] = data['KitchenQual_num'] * data['GrLivArea']

    # 7. Общ брой бани (по-прецизно)
    data['TotalBath_precise'] = data['FullBath'] + 0.5 * data['HalfBath']
    if 'BsmtFullBath' in data.columns:
        data['TotalBath_precise'] += data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']

    # 8. Външно качество
    if 'ExterQual' in data.columns:
        exter_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
        data['ExterQual_num'] = data['ExterQual'].map(exter_map).fillna(0)

    # 9. Скъпи материали
    expensive_materials = ['BrkFace', 'Stone', 'BrkComm', 'CBlock']
    if 'Exterior1st' in data.columns:
        data['ExpensiveMaterial'] = data['Exterior1st'].isin(expensive_materials).astype(int)

    # 10. Време от последния ремонт (ако има)
    if 'YearRemodAdd' in data.columns and 'YearBuilt' in data.columns:
        data['RemodAge'] = data['YearRemodAdd'] - data['YearBuilt']
        data['RemodAge'] = data['RemodAge'].clip(0, 50)

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

# 6. ПОДОБРЕН STACKING МОДЕЛ (5 модела + Ridge като мета-модел)
print("\n6. Създаване на Stacking модел...")

# ПОДОБРЕНИ ПАРАМЕТРИ
rf_params = {'n_estimators': 500, 'max_depth': 25, 'min_samples_split': 5,
             'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1}

xgb_params = {'n_estimators': 1000, 'learning_rate': 0.02, 'max_depth': 6,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
              'verbosity': 0}

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,  # увеличено от 31
    'learning_rate': 0.02,  # намалено от 0.05
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'n_estimators': 1500,
    'verbose': -1,
    'random_state': 42
}

# НОВО: Добавяме Gradient Boosting
gb_params = {'n_estimators': 500, 'learning_rate': 0.02, 'max_depth': 5,
             'subsample': 0.8, 'random_state': 42}

# НОВО: Добавяме Ridge Regression
ridge_params = {'alpha': 1.0, 'random_state': 42}

# Кръстосана валидация
print("\n   Кръстосана валидация (5-fold)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Масиви за съхранение на прогнозите (5 модела)
oof_rf = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_gb = np.zeros(len(X_train))
oof_ridge = np.zeros(len(X_train))

test_preds_rf = np.zeros((len(X_test), 5))
test_preds_xgb = np.zeros((len(X_test), 5))
test_preds_lgb = np.zeros((len(X_test), 5))
test_preds_gb = np.zeros((len(X_test), 5))
test_preds_ridge = np.zeros((len(X_test), 5))

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
    xg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xg.predict(X_val)
    test_preds_xgb[:, fold] = xg.predict(X_test)

    # LightGBM
    lg = lgb.LGBMRegressor(**lgb_params)
    lg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[val_idx] = lg.predict(X_val)
    test_preds_lgb[:, fold] = lg.predict(X_test)

    # Gradient Boosting (НОВ)
    gb = GradientBoostingRegressor(**gb_params)
    gb.fit(X_tr, y_tr)
    oof_gb[val_idx] = gb.predict(X_val)
    test_preds_gb[:, fold] = gb.predict(X_test)

    # Ridge Regression (НОВ)
    from sklearn.linear_model import Ridge
    ridge = Ridge(**ridge_params)
    ridge.fit(X_tr, y_tr)
    oof_ridge[val_idx] = ridge.predict(X_val)
    test_preds_ridge[:, fold] = ridge.predict(X_test)

# Средни прогнози за тест
test_preds_rf_mean = test_preds_rf.mean(axis=1)
test_preds_xgb_mean = test_preds_xgb.mean(axis=1)
test_preds_lgb_mean = test_preds_lgb.mean(axis=1)
test_preds_gb_mean = test_preds_gb.mean(axis=1)
test_preds_ridge_mean = test_preds_ridge.mean(axis=1)

# 7. Meta Model (Ridge Regression - по-стабилен от LinearRegression)
print("\n7. Обучение на Meta модел (Ridge Regression)...")
from sklearn.linear_model import Ridge

meta_train = pd.DataFrame({
    'rf': oof_rf,
    'xgb': oof_xgb,
    'lgb': oof_lgb,
    'gb': oof_gb,
    'ridge': oof_ridge
})

meta_test = pd.DataFrame({
    'rf': test_preds_rf_mean,
    'xgb': test_preds_xgb_mean,
    'lgb': test_preds_lgb_mean,
    'gb': test_preds_gb_mean,
    'ridge': test_preds_ridge_mean
})

# НОВО: Ridge като мета-модел (по-стабилен)
meta_model = Ridge(alpha=0.02, random_state=42)
meta_model.fit(meta_train, y_train)

# 8. ПРОВЕРКА НА МЕТА МОДЕЛА
print("\n   ✓ Meta модел коефициенти:")
for model_name, coef in zip(['rf', 'xgb', 'lgb', 'gb', 'ridge'], meta_model.coef_):
    print(f"      {model_name}: {coef:.3f}")

# Прогнози
final_predictions = meta_model.predict(meta_test)

# 9. СЪЗДАВАНЕ НА CSV
print("\n8. Създаване на CSV файл...")
predictions = np.expm1(final_predictions)

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': predictions
})
submission.to_csv('submission_house_prices_improved_v_06.csv', index=False)

print(f"\n✅ Файл: submission_house_prices_improved_v_06.csv")
print(f"\n📊 СТАТИСТИКА НА ПРОГНОЗИТЕ:")
print(f"   Минимална: ${predictions.min():,.0f}")
print(f"   Максимална: ${predictions.max():,.0f}")
print(f"   Средна: ${predictions.mean():,.0f}")

# 10. СРАВНЕНИЕ
print("\n" + "=" * 80)
print("📈 ФИНАЛНО СРАВНЕНИЕ НА ВСИЧКИ МОДЕЛИ:")
print("=" * 80)
print(f"   1. Random Forest (11 features)        : 0.15793")
print(f"   2. XGBoost (70+ features)             : 0.12854")
print(f"   3. LightGBM (80+ features)            : 0.12505")
print(f"   4. Вашият Stacking (3 модела)         : 0.12415")
print(f"   5. ПОДОБРЕН Stacking (5 модела)       : ЦЕЛ 0.121 - 0.123")
print("=" * 80)

print("\n💡 ПРОГНОЗА ЗА KAGGLE:")
print("   🎉 ОЧАКВАМ 0.121 - 0.123!")
print("   Подобрения: 5 модела + по-добър meta-learner + нови features")

print("\n🚀 ГОТОВО ЗА КАЧВАНЕ В KAGGLE!")
print("📁 Файл: submission_house_prices_improved_v_06.csv")


