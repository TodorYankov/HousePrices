# test_feature_selection_lasso.py
# Тества дали премахването на маловажни признаци (чрез Lasso) подобрява резултата

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb

print("=" * 80)
print("🔬 ТЕСТ: FEATURE SELECTION С LASSO")
print("=" * 80)

# ============================================================================
# 1. ЗАРЕЖДАНЕ И ПРЕПРОЦЕСИНГ (както в v_06)
# ============================================================================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

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

train = fix_missing_values(train)
test = fix_missing_values(test)
train, neigh_median_dict = add_features(train)
test, _ = add_features(test, neigh_median=neigh_median_dict)

cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col != 'SalePrice':
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

feature_cols = [c for c in train.columns if c not in ['Id', 'SalePrice']]
X = train[feature_cols].fillna(0)
y = np.log1p(train['SalePrice'])

outlier_cond = (X['GrLivArea'] > 4000) & (np.expm1(y) < 300000)
outlier_cond |= (X['TotalBsmtSF'] > 3000) & (np.expm1(y) < 200000)
X = X[~outlier_cond]
y = y[~outlier_cond]
print(f"Премахнати аутлайъри: {outlier_cond.sum()}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test = test[feature_cols].fillna(0)
X_test_scaled = scaler.transform(X_test)

print(f"Първоначален брой признаци: {X_scaled.shape[1]}")

# ============================================================================
# 2. FEATURE SELECTION С LASSO
# ============================================================================
print("\n📊 Избор на признаци с LassoCV...")
lasso = LassoCV(cv=5, random_state=42, max_iter=10000, n_alphas=100)
lasso.fit(X_scaled, y)

# Избрани признаци (ненулеви коефициенти)
selected_mask = lasso.coef_ != 0
selected_features = np.array(feature_cols)[selected_mask]
X_selected = X_scaled[:, selected_mask]
X_test_selected = X_test_scaled[:, selected_mask]

print(f"Избрани признаци: {len(selected_features)} от {X_scaled.shape[1]}")
print(f"Брой ненулеви коефициенти: {np.sum(selected_mask)}")

# ============================================================================
# 3. ОЦЕНКА НА МОДЕЛА С И БЕЗ SELECTION
# ============================================================================
print("\n🔬 Сравнение на CV RMSLE (5-fold) с и без feature selection:")

# Параметри (същите като v_06)
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

models = [
    ('rf', RandomForestRegressor(**rf_params)),
    ('xgb', xgb.XGBRegressor(**xgb_params)),
    ('lgb', lgb.LGBMRegressor(**lgb_params)),
    ('gb', GradientBoostingRegressor(**gb_params)),
    ('ridge', Ridge(**ridge_params))
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(X_data, y_data, models, kf):
    oof = np.zeros((len(X_data), len(models)))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_data)):
        X_tr, X_val = X_data[tr_idx], X_data[val_idx]
        y_tr, y_val = y_data.iloc[tr_idx], y_data.iloc[val_idx]
        for i, (name, model) in enumerate(models):
            model.fit(X_tr, y_tr)
            oof[val_idx, i] = model.predict(X_val)
    # Meta model (Ridge alpha=0.02)
    meta = Ridge(alpha=0.02, random_state=42)
    meta.fit(oof, y_data)
    pred = meta.predict(oof)
    pred = np.maximum(pred, 0)
    rmsle = np.sqrt(mean_squared_log_error(y_data, pred))
    return rmsle, meta

# Оценка с всички признаци
rmsle_full, _ = evaluate_model(X_scaled, y, models, kf)
print(f"   С всички {X_scaled.shape[1]} признаци: RMSLE = {rmsle_full:.5f}")

# Оценка с избраните признаци
rmsle_selected, _ = evaluate_model(X_selected, y, models, kf)
print(f"   Само с {X_selected.shape[1]} избрани признаци: RMSLE = {rmsle_selected:.5f}")

improvement = rmsle_full - rmsle_selected
print("\n" + "=" * 80)
if improvement > 0:
    print(f"✅ Feature selection подобрява резултата с {improvement:.5f} (по-нисък RMSLE)")
    print("💡 ПРЕПОРЪКА: Използвайте избраните признаци във финалния модел.")
else:
    print(f"❌ Feature selection не подобрява резултата (разлика {improvement:.5f})")
    print("💡 ПРЕПОРЪКА: Останете с всички признаци (v_06).")
print("=" * 80)