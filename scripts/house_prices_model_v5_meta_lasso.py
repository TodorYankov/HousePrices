# house_prices_model_v5_meta_lasso.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("HOUSE PRICES MODEL V5 - META LASSO (микро подобрение)")
print("=" * 80)

# ============================================================
# 1. ЗАРЕЖДАНЕ И ПОДГОТОВКА (ТОЧНО КАТО В НАЙ-ДОБРИЯ МОДЕЛ)
# ============================================================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


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
    data['TotalBath'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']
    if 'YearBuilt' in data.columns:
        data['HouseAge'] = 2010 - data['YearBuilt']
    data['Qual_Area'] = data['OverallQual'] * data['GrLivArea']
    data['Qual_TotalSF'] = data['OverallQual'] * data['TotalSF']
    data['HasBsmt'] = (data['TotalBsmtSF'] > 0).astype(int)
    data['HasGarage'] = (data['GarageArea'] > 0).astype(int)
    data['TopNeighborhood'] = data['Neighborhood'].isin(['StoneBr', 'NridgHt', 'NoRidge']).astype(int)
    if 'MoSold' in data.columns:
        data['Spring'] = data['MoSold'].isin([3, 4, 5]).astype(int)
        data['Summer'] = data['MoSold'].isin([6, 7, 8]).astype(int)
        data['Fall'] = data['MoSold'].isin([9, 10, 11]).astype(int)
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    if 'KitchenQual' in data.columns:
        data['KitchenQual_num'] = data['KitchenQual'].map(qual_map).fillna(0)
    if 'BsmtQual' in data.columns:
        data['BsmtQual_num'] = data['BsmtQual'].map(qual_map).fillna(0)
    if 'ExterQual' in data.columns:
        data['ExterQual_num'] = data['ExterQual'].map(qual_map).fillna(0)
    return data


train = fix_missing_values(train)
test = fix_missing_values(test)
train = add_features(train)
test = add_features(test)

cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

feature_cols = [col for col in train.columns if col not in ['Id', 'SalePrice']]
X = train[feature_cols].fillna(0)
y = np.log1p(train['SalePrice'])
X_test = test[feature_cols].fillna(0)

print(f"Брой features: {X.shape[1]}")

# ============================================================
# 2. МОДЕЛИ (същите като в improved модела)
# ============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
xgb = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, random_state=42, verbosity=0)
lgb = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

models = [('rf', rf), ('xgb', xgb), ('lgb', lgb), ('gb', gb)]

# OOF прогнози
oof_preds = {name: np.zeros(len(X)) for name, _ in models}
test_preds = {name: np.zeros((len(X_test), 5)) for name, _ in models}

print("\n5-fold CV stacking...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"   Fold {fold + 1}/5")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    for name, model in models:
        model.fit(X_tr, y_tr)
        oof_preds[name][val_idx] = model.predict(X_val)
        test_preds[name][:, fold] = model.predict(X_test)

# Мета features
meta_train = pd.DataFrame({name: oof_preds[name] for name, _ in models})
meta_test = pd.DataFrame({name: test_preds[name].mean(axis=1) for name, _ in models})

# ============================================================
# 3. ЕДИНСТВЕНАТА ПРОМЯНА: МЕТА МОДЕЛ Lasso (вместо Ridge)
# ============================================================
meta_model = Lasso(alpha=0.001, max_iter=5000, random_state=42)
meta_model.fit(meta_train, y)

print("\nКоефициенти на мета модела (Lasso):")
for name, coef in zip([name for name, _ in models], meta_model.coef_):
    print(f"   {name}: {coef:.5f}")
nonzero = np.sum(meta_model.coef_ != 0)
print(f"Ненулеви коефициенти: {nonzero} от {len(models)}")

primary_pred = meta_model.predict(meta_test)

# ============================================================
# 4. ФИНАЛНА ПРОГНОЗА
# ============================================================
final_predictions = np.expm1(primary_pred)

submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': final_predictions})
submission.to_csv('house_prices_model_v5_meta_lasso.csv', index=False)

print("\n" + "=" * 80)
print("📊 РЕЗУЛТАТИ")
print("=" * 80)
print(f"   Мета модел: Lasso (alpha=0.001)")
print(f"   Брой ненулеви коефициенти: {nonzero}")
print(f"   Статистика на прогнозите:")
print(f"   Минимална: ${final_predictions.min():,.0f}")
print(f"   Максимална: ${final_predictions.max():,.0f}")
print(f"   Средна: ${final_predictions.mean():,.0f}")

print("\n🚀 ГОТОВО ЗА КАЧВАНЕ!")
print("📁 Файл: house_prices_model_v5_meta_lasso.csv")
print("=" * 80)