# house_prices_h2o_full.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("HOUSE PRICES - H2O-3 AutoML (2 часа + XGBoost + ансамбъл)")
print("=" * 80)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print(f"\nTrain: {train.shape}, Test: {test.shape}")

# --- Функции за обработка (същите като v_06) ---
def fix_missing_values(df):
    data = df.copy()
    if 'LotFrontage' in data.columns and 'Neighborhood' in data.columns:
        data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
    num_features = ['MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2',
                    'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                    'GarageArea', 'GarageCars']
    for col in num_features:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
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

def add_features(df):
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
        data['Spring'] = data['MoSold'].isin([3,4,5]).astype(int)
        data['Summer'] = data['MoSold'].isin([6,7,8]).astype(int)
        data['Fall'] = data['MoSold'].isin([9,10,11]).astype(int)
    data['QualPerAge'] = data['OverallQual'] / (data['HouseAge'] + 1)
    data['TotalRooms'] = data['TotRmsAbvGrd'] + data['BsmtFinSF1'] / 500
    data['IsUrban'] = data['MSSubClass'].isin([60, 120, 150, 180, 190]).astype(int)
    kitchen_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0}
    if 'KitchenQual' in data.columns:
        data['KitchenQual_num'] = data['KitchenQual'].map(kitchen_map).fillna(0)
    bsmt_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0}
    if 'BsmtQual' in data.columns:
        data['BsmtQual_num'] = data['BsmtQual'].map(bsmt_map).fillna(0)
    if 'KitchenQual_num' in data.columns:
        data['Kitchen_Area'] = data['KitchenQual_num'] * data['GrLivArea']
    data['TotalBath_precise'] = data['FullBath'] + 0.5*data['HalfBath']
    if 'BsmtFullBath' in data.columns:
        data['TotalBath_precise'] += data['BsmtFullBath'] + 0.5*data['BsmtHalfBath']
    if 'ExterQual' in data.columns:
        exter_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}
        data['ExterQual_num'] = data['ExterQual'].map(exter_map).fillna(0)
    expensive_materials = ['BrkFace', 'Stone', 'BrkComm', 'CBlock']
    if 'Exterior1st' in data.columns:
        data['ExpensiveMaterial'] = data['Exterior1st'].isin(expensive_materials).astype(int)
    if 'YearRemodAdd' in data.columns and 'YearBuilt' in data.columns:
        data['RemodAge'] = data['YearRemodAdd'] - data['YearBuilt']
        data['RemodAge'] = data['RemodAge'].clip(0, 50)
    return data

print("\nПрилагане на обработка...")
train = fix_missing_values(train)
test = fix_missing_values(test)
train = add_features(train)
test = add_features(test)

from sklearn.preprocessing import LabelEncoder
cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col in train.columns and col != 'SalePrice':
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

feature_cols = [col for col in train.columns if col not in ['Id', 'SalePrice']]
X_train = train[feature_cols].fillna(0)
y_train = np.log1p(train['SalePrice'])
X_test = test[feature_cols].fillna(0)
print(f"Брой признаци: {len(feature_cols)}")

# ============================================================
# H2O-3 AutoML (пълен експеримент)
# ============================================================
print("\nСтартиране на H2O-3...")
import h2o
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame
import os

os.environ['JAVA_HOME'] = "C:/Program Files/Eclipse Adoptium/jdk-21.0.11.10-hotspot"
os.environ['PATH'] = os.environ['JAVA_HOME'] + "/bin;" + os.environ['PATH']

# НАЙ-ВАЖНАТА ПРОМЯНА – премахнат enable_algos=None
h2o.init(max_mem_size="4G")

# Създаване на H2O фреймове
train_df = X_train.copy()
train_df['SalePrice_log'] = y_train
train_h2o = H2OFrame(train_df)
test_h2o = H2OFrame(X_test)
target = 'SalePrice_log'

print("\nСтартиране на AutoML (2 часа, 50 модела, 5-кратна CV)...")
aml = H2OAutoML(
    max_runtime_secs=7200,      # 2 часа
    max_models=50,
    seed=42,
    sort_metric='RMSE',
    nfolds=5,
    verbosity='info'
)
aml.train(y=target, training_frame=train_h2o)

lb = aml.leaderboard
print("\n📊 Leaderboard (топ 10):")
print(lb.head(10))

best = aml.leader
best_rmsle = lb[0, 'rmse']
print(f"\n🏆 Най-добър модел: {best.model_id}")
print(f"   CV RMSLE: {best_rmsle:.5f}")

# ============================================================
# Прогнози
# ============================================================
print("\nГенериране на прогнози...")
preds = best.predict(test_h2o)
preds_log = preds.as_data_frame().iloc[:, 0]
preds_price = np.expm1(preds_log)

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': preds_price
})
submission.to_csv('submission_h2o_autoML_full.csv', index=False)
print("\n✅ Файл: submission_h2o_autoML_full.csv")

# ============================================================
# Ансамбъл с v_06
# ============================================================
print("\nСъздаване на ансамбъл с вашия стек (v_06)...")
old = pd.read_csv('submission_house_prices_improved_v_06.csv')
old_preds = old['SalePrice']
ensemble_preds = 0.5 * old_preds + 0.5 * preds_price

ensemble_sub = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': ensemble_preds
})
ensemble_sub.to_csv('submission_ensemble_h2o_v06.csv', index=False)
print("✅ Файл: submission_ensemble_h2o_v06.csv")

# ============================================================
# Сравнение
# ============================================================
print("\n" + "=" * 80)
print("СРАВНЕНИЕ")
print("=" * 80)
print(f"   Вашият Stacking (v_06)      : 0.11974  (CV RMSLE)")
print(f"   H2O-3 AutoML (2 часа)      : {best_rmsle:.5f}  (CV RMSLE)")
print(f"   Ансамбъл (v_06 + H2O)      : качете в Kaggle за проверка")

if best_rmsle < 0.11974:
    print("\n🎉 H2O-3 AutoML ПОПРАВИ резултата!")
else:
    print("\n⚠️ H2O-3 не подобри самостоятелно, но ансамбълът може да даде резултат.")
print("=" * 80)

h2o.cluster().shutdown()
