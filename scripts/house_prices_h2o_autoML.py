# house_prices_h2o_autoML.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. ЗАРЕЖДАНЕ И ОБРАБОТКА НА ДАННИ (като във v_06)
# ============================================================
print("=" * 80)
print("HOUSE PRICES - H2O-3 AutoML (опит за подобрение)")
print("=" * 80)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print(f"\nTrain: {train.shape}, Test: {test.shape}")

# Функции за обработка (копирани от v_06)
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

# Кодиране на категорийни променливи
from sklearn.preprocessing import LabelEncoder
cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col in train.columns and col != 'SalePrice':
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

# Подготовка на данните
feature_cols = [col for col in train.columns if col not in ['Id', 'SalePrice']]
X_train = train[feature_cols].fillna(0)
y_train = np.log1p(train['SalePrice'])   # логаритмувана цел
X_test = test[feature_cols].fillna(0)
print(f"Брой признаци: {len(feature_cols)}")

# ============================================================
# 2. H2O-3 AutoML
# ============================================================
print("\nСтартиране на H2O-3...")
import h2o
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame
import os

# Задаваме JAVA_HOME ръчно (за да може H2O да намери Java)
os.environ['JAVA_HOME'] = "C:/Program Files/Eclipse Adoptium/jdk-21.0.11.10-hotspot"
os.environ['PATH'] = os.environ['JAVA_HOME'] + "/bin;" + os.environ['PATH']

# Стартиране на H2O с 4GB памет
h2o.init(max_mem_size="4G")

# Конвертиране към H2OFrame – по-чист начин
train_df = X_train.copy()
train_df['SalePrice_log'] = y_train
train_h2o = H2OFrame(train_df)
test_h2o = H2OFrame(X_test)
target = 'SalePrice_log'

print("\nСтартиране на AutoML (15 минути за бърз тест)...")
aml = H2OAutoML(
    max_runtime_secs=900,      # 15 минути – за бърз тест (сменете на 7200 за 2 часа)
    max_models=20,
    seed=42,
    sort_metric='RMSE',
    nfolds=3,
    verbosity='info'
)
aml.train(y=target, training_frame=train_h2o)

# Взимаме най-добрия модел и leaderboard
lb = aml.leaderboard
best = aml.leader

print("\n📊 Leaderboard (топ 10):")
print(lb.head(10))
print(f"\n🏆 Най-добър модел: {best.model_id}")
print(f"   CV RMSLE: {lb[0, 'rmse']:.5f}")

# ============================================================
# 3. Прогнози върху тестови данни
# ============================================================
print("\nГенериране на прогнози...")
preds = best.predict(test_h2o)
preds_log = preds.as_data_frame().iloc[:, 0]   # прогнози в лог скала
preds_price = np.expm1(preds_log)              # обратно трансформиране

# Запазване
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': preds_price
})
submission.to_csv('submission_h2o_autoML.csv', index=False)
print("\n✅ Файл: submission_h2o_autoML.csv")

# ============================================================
# 4. Сравнение с вашия най-добър резултат
# ============================================================
print("\n" + "=" * 80)
print("СРАВНЕНИЕ")
print("=" * 80)
print(f"   Вашият Stacking (v_06) : 0.11974  (CV RMSLE)")
print(f"   H2O-3 AutoML (best)   : {lb[0, 'rmse']:.5f}  (CV RMSLE)")

if lb[0, 'rmse'] < 0.11974:
    print("\n🎉 H2O-3 AutoML ПОПРАВИ резултата! Очаквайте по-добро класиране в Kaggle.")
else:
    print("\n⚠️ H2O-3 не подобри резултата. Може да опитате с по-дълго време или да комбинирате двата модела.")
print("=" * 80)

# ============================================================
# 5. (Опционално) Ансамбъл между вашия стек и H2O
# ============================================================
# Ако искате да опитате комбинация, можете да заредите вашите стари прогнози
# и да направите средно аритметично или претеглено средно с H2O прогнозите.
# Пример:
# old_preds = pd.read_csv('submission_house_prices_improved_v_06.csv')['SalePrice']
# ensemble_preds = 0.5 * old_preds + 0.5 * preds_price
# и запишете като 'submission_ensemble_h2o.csv'
