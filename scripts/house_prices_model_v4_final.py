# house_prices_model_v4_final.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("HOUSE PRICES MODEL V4 - FINAL TUNING (Target: 0.118-0.119)")
print("=" * 80)

# ============================================================
# 1. ЗАРЕЖДАНЕ
# ============================================================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")


# ============================================================
# 2. ОБРАБОТКА НА ЛИПСВАЩИ СТОЙНОСТИ (както в най-добрия модел)
# ============================================================
def fix_missing_values(df):
    data = df.copy()
    # Numerical features
    num_features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1',
                    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
                    'BsmtHalfBath', 'GarageArea', 'GarageCars']
    for col in num_features:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    # Categorical features
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


# ============================================================
# 3. FEATURE ENGINEERING (вашите доказани features)
# ============================================================
def add_features(df):
    data = df.copy()
    data['TotalSF'] = data['TotalBsmtSF'] + data['GrLivArea']
    data['TotalBath'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']
    if 'YearBuilt' in data.columns:
        data['HouseAge'] = 2010 - data['YearBuilt']
    if 'YearRemodAdd' in data.columns:
        data['YearsSinceRemod'] = 2010 - data['YearRemodAdd']
    data['Qual_Area'] = data['OverallQual'] * data['GrLivArea']
    data['Qual_TotalSF'] = data['OverallQual'] * data['TotalSF']
    data['HasBsmt'] = (data['TotalBsmtSF'] > 0).astype(int)
    data['HasGarage'] = (data['GarageArea'] > 0).astype(int)
    data['HasFireplace'] = (data['Fireplaces'] > 0).astype(int)
    # Топ квартали
    top_neighborhoods = ['StoneBr', 'NridgHt', 'NoRidge']
    data['TopNeighborhood'] = data['Neighborhood'].isin(top_neighborhoods).astype(int)
    # Сезонност
    if 'MoSold' in data.columns:
        data['Spring'] = data['MoSold'].isin([3, 4, 5]).astype(int)
        data['Summer'] = data['MoSold'].isin([6, 7, 8]).astype(int)
        data['Fall'] = data['MoSold'].isin([9, 10, 11]).astype(int)
    # Качество като числа (ако не са вече)
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    if 'KitchenQual' in data.columns:
        data['KitchenQual_num'] = data['KitchenQual'].map(qual_map).fillna(0)
    if 'BsmtQual' in data.columns:
        data['BsmtQual_num'] = data['BsmtQual'].map(qual_map).fillna(0)
    if 'ExterQual' in data.columns:
        data['ExterQual_num'] = data['ExterQual'].map(qual_map).fillna(0)
    # Брой стаи (приблизително)
    data['TotRooms'] = data['TotRmsAbvGrd'] + data['BsmtFinSF1'] / 500
    return data


train = add_features(train)
test = add_features(test)

# ============================================================
# 4. КОДИРАНЕ НА КАТЕГОРИЙНИТЕ ПРОМЕНЛИВИ
# ============================================================
cat_cols = train.select_dtypes(include=['object']).columns
print(f"Категорийни колони: {len(cat_cols)}")
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ============================================================
# 5. ПОДГОТОВКА НА X, y, X_test
# ============================================================
feature_cols = [col for col in train.columns if col not in ['Id', 'SalePrice']]
X = train[feature_cols].fillna(0)
y = np.log1p(train['SalePrice'])  # Логаритмична трансформация – стабилна и доказана
X_test = test[feature_cols].fillna(0)

print(f"Брой features преди селекция: {X.shape[1]}")

# ============================================================
# 6. НОРМАЛИЗАЦИЯ (за Lasso и Ridge)
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 7. FEATURE SELECTION С LASSO (агресивно – alpha=0.01)
# ============================================================
lasso = Lasso(alpha=0.01, max_iter=5000, random_state=42)
lasso.fit(X_scaled, y)
selected = lasso.coef_ != 0
selected_features = X.columns[selected]
print(f"Избрани features след Lasso: {len(selected_features)}")

X = X[selected_features]
X_test = X_test[selected_features]
X_scaled = scaler.fit_transform(X)  # повторно скалиране (само за избраните)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 8. OUTLIER REMOVAL (Z-score на target)
# ============================================================
z_scores = np.abs((y - y.mean()) / y.std())
outliers = z_scores > 3
print(f"Премахнати outliers: {outliers.sum()}")
X_scaled = X_scaled[~outliers]
X = X.iloc[~outliers.values]  # за запазване на оригиналния DataFrame (ако е нужен)
y = y[~outliers]

# ============================================================
# 9. STACKING С 6 МОДЕЛА (RF, XGB, LGB, GB, CatBoost, Ridge)
# ============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Модели
rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
xgb = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, random_state=42, verbosity=0)
lgb = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
cat = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=5, random_seed=42, verbose=False)
ridge = Ridge(alpha=0.5, random_state=42)

models = [('rf', rf), ('xgb', xgb), ('lgb', lgb), ('gb', gb), ('cat', cat), ('ridge', ridge)]

# Масиви за OOF и тестови прогнози
oof_preds = {name: np.zeros(len(X_scaled)) for name, _ in models}
test_preds = {name: np.zeros((len(X_test_scaled), 5)) for name, _ in models}

print("\nОбучение на stacking (5-fold CV):")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
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
meta_model = Ridge(alpha=0.3, random_state=42)
meta_model.fit(meta_train, y)
primary_pred = meta_model.predict(meta_test)

print("\nКоефициенти на мета модела:")
for name, coef in zip([name for name, _ in models], meta_model.coef_):
    print(f"   {name}: {coef:.3f}")

# ============================================================
# 10. 2-STAGE PREDICTION (RF на residuals)
# ============================================================
train_pred = meta_model.predict(meta_train)
residuals = y - train_pred

rf_res = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
rf_res.fit(meta_train, residuals)
residuals_pred = rf_res.predict(meta_test)

final_predictions_log = primary_pred + residuals_pred
final_predictions = np.expm1(final_predictions_log)

# ============================================================
# 11. СТАТИСТИКА И ЗАПИС
# ============================================================
print("\n" + "=" * 80)
print("📊 РЕЗУЛТАТИ")
print("=" * 80)
print(f"   Брой features след Lasso: {len(selected_features)}")
print(f"   Модели в ансамбъла: RF, XGB, LGB, GB, CatBoost, Ridge")
print(f"   2-stage: RF на residuals")

print(f"\n📊 Статистика на прогнозите:")
print(f"   Минимална: ${final_predictions.min():,.0f}")
print(f"   Максимална: ${final_predictions.max():,.0f}")
print(f"   Средна: ${final_predictions.mean():,.0f}")

# Запис
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': final_predictions})
submission.to_csv('house_prices_model_v4_final.csv', index=False)

print("\n🚀 ГОТОВО ЗА КАЧВАНЕ!")
print("📁 Файл: house_prices_model_v4_final.csv")
print("=" * 80)