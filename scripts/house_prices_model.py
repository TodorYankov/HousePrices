# house_prices_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("HOUSE PRICES - RANDOM FOREST МОДЕЛ")
print("=" * 80)

# 1. ЗАРЕЖДАНЕ НА ДАННИТЕ
print("\n1. Зареждане на данни...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"   Train: {train.shape}")
print(f"   Test: {test.shape}")

# 2. ПРОСТА ПОДГОТОВКА НА ДАННИ
print("\n2. Подготовка на данни...")

# Избираме най-важните числови features
features = [
    'OverallQual',      # Качество (1-10)
    'GrLivArea',        # Жилищна площ
    'GarageCars',       # Брой коли в гараж
    'GarageArea',       # Площ на гараж
    'TotalBsmtSF',      # Обща площ на мазе
    '1stFlrSF',         # Площ на първи етаж
    'YearBuilt',        # Година на строителство
    'YearRemodAdd',     # Година на ремонт
    'LotArea',          # Площ на парцела
    'FullBath',         # Брой бани
    'BedroomAbvGr'      # Брой спални
]

# Проверка кои features съществуват
available_features = [f for f in features if f in train.columns]
print(f"   Използвани features: {available_features}")

# Запълване на липсващи стойности с 0
X_train = train[available_features].fillna(0)
X_test = test[available_features].fillna(0)

# Целева променлива - логаритмуваме за по-добро разпределение
y_train = np.log1p(train['SalePrice'])

# Стандартизиране
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. МОДЕЛ
print("\n3. Обучение на Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Кръстосана валидация
cv_scores = cross_val_score(rf, X_train_scaled, y_train,
                             cv=5, scoring='neg_root_mean_squared_error')
print(f"   CV RMSLE: {-cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# 4. ОБУЧАВАНЕ НА ФИНАЛЕН МОДЕЛ
print("\n4. Обучение на финален модел...")
rf.fit(X_train_scaled, y_train)

# 5. ПРОГНОЗИРАНЕ
print("\n5. Прогнозиране...")
predictions = np.expm1(rf.predict(X_test_scaled))

# 6. СЪЗДАВАНЕ НА SUBMISSION
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': predictions
})
submission.to_csv('submission_house_prices.csv', index=False)

print(f"\n✅ Файл: submission_house_prices.csv")
print(f"📊 Статистика на прогнозите:")
print(f"   Минимална: ${predictions.min():,.0f}")
print(f"   Максимална: ${predictions.max():,.0f}")
print(f"   Средна: ${predictions.mean():,.0f}")
print(f"   Медианна: ${np.median(predictions):,.0f}")

print("\n🚀 ГОТОВО ЗА КАЧВАНЕ В KAGGLE!")
