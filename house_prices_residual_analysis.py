# house_prices_residual_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Зареждане и подготовка (същата като най-добрия модел)
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
    data['TotalBath'] = data['FullBath'] + 0.5*data['HalfBath'] + data['BsmtFullBath'] + 0.5*data['BsmtHalfBath']
    if 'YearBuilt' in data.columns:
        data['HouseAge'] = 2010 - data['YearBuilt']
    data['Qual_Area'] = data['OverallQual'] * data['GrLivArea']
    data['Qual_TotalSF'] = data['OverallQual'] * data['TotalSF']
    data['HasBsmt'] = (data['TotalBsmtSF'] > 0).astype(int)
    data['HasGarage'] = (data['GarageArea'] > 0).astype(int)
    data['TopNeighborhood'] = data['Neighborhood'].isin(['StoneBr', 'NridgHt', 'NoRidge']).astype(int)
    if 'MoSold' in data.columns:
        data['Spring'] = data['MoSold'].isin([3,4,5]).astype(int)
        data['Summer'] = data['MoSold'].isin([6,7,8]).astype(int)
        data['Fall'] = data['MoSold'].isin([9,10,11]).astype(int)
    qual_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0}
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

# ============================================================
# 2. Обучение на най-добрия модел (5-моделен stacking с Ridge alpha=1.0)
# ============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
xgb = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, random_state=42, verbosity=0)
lgb = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

models = [('rf', rf), ('xgb', xgb), ('lgb', lgb), ('gb', gb)]

oof_preds = {name: np.zeros(len(X)) for name, _ in models}
test_preds = {name: np.zeros((len(X_test), 5)) for name, _ in models}

print("5-fold CV stacking...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"   Fold {fold+1}/5")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    for name, model in models:
        model.fit(X_tr, y_tr)
        oof_preds[name][val_idx] = model.predict(X_val)
        test_preds[name][:, fold] = model.predict(X_test)

meta_train = pd.DataFrame({name: oof_preds[name] for name, _ in models})
meta_test = pd.DataFrame({name: test_preds[name].mean(axis=1) for name, _ in models})

meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(meta_train, y)

train_pred = meta_model.predict(meta_train)
residuals = y - train_pred

# ============================================================
# 3. Анализ на residuals
# ============================================================
print("\n" + "="*80)
print("АНАЛИЗ НА ГРЕШКИТЕ (RESIDUALS)")
print("="*80)

abs_residuals = np.abs(residuals)
print(f"Средна абсолютна грешка (log scale): {abs_residuals.mean():.4f}")
print(f"Медианна абсолютна грешка: {np.median(abs_residuals):.4f}")

# Най-големи грешки
top_errors_idx = np.argsort(abs_residuals)[-10:]
print("\nТоп 10 най-големи грешки (индекси):", top_errors_idx)

# Връзка с оригиналните данни
errors_df = train.iloc[top_errors_idx].copy()
errors_df['True_Price'] = train['SalePrice'].iloc[top_errors_idx]
errors_df['Pred_Price'] = np.expm1(train_pred[top_errors_idx])
errors_df['Residual'] = residuals.iloc[top_errors_idx]
print("\nХарактеристики на най-грешните прогнози:")
print(errors_df[['Neighborhood', 'OverallQual', 'GrLivArea', 'TotalSF', 'True_Price', 'Pred_Price']].to_string())

# Грешки по квартали
residuals_by_neigh = pd.DataFrame({'Neighborhood': train['Neighborhood'], 'Residual': residuals})
grouped = residuals_by_neigh.groupby('Neighborhood')['Residual'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
print("\nКвартали с най-голяма средна грешка (положителна = подценяване):")
print(grouped.head(10))

# Грешки по клас на къщата (OverallQual)
print("\nГрешки по OverallQual:")
for q in sorted(train['OverallQual'].unique()):
    q_res = residuals[train['OverallQual'] == q]
    if len(q_res) > 0:
        print(f"  Qual {q}: mean residual = {q_res.mean():.4f} (count={len(q_res)})")

# Грешки по сегмент на цената
price_segments = pd.cut(train['SalePrice'], bins=[0, 150000, 250000, 600000], labels=['Low', 'Medium', 'High'])
for seg in ['Low', 'Medium', 'High']:
    seg_res = residuals[price_segments == seg]
    if len(seg_res) > 0:
        print(f"  {seg} price: mean residual = {seg_res.mean():.4f} (std={seg_res.std():.4f})")