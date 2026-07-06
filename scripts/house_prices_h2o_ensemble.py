# house_prices_h2o_ensemble.py
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 80)
print("HOUSE PRICES - H2O-3 AutoML + Ансамбъл с вашия най-добър модел")
print("=" * 80)

# 1. Зареждане на H2O прогнозите (вече са генерирани)
h2o_file = 'submission_h2o_autoML_full.csv'
if not os.path.exists(h2o_file):
    print(f"Грешка: Файл {h2o_file} не е намерен. Първо стартирайте основния скрипт.")
    exit()

h2o_preds = pd.read_csv(h2o_file)
print(f"Заредени H2O прогнози: {h2o_file}")

# 2. Търсене на най-добрия ви CSV файл
best_files = [
    'submission_house_prices_improved_v_06.csv',
    'submission_house_prices_improved_v_07.csv',
    'submission_house_prices_improved_v_05.csv',
    'submission_house_prices_improved.csv',
    'submission_house_prices_stacking.csv'
]

old_preds = None
used_file = None

for fname in best_files:
    if os.path.exists(fname):
        old_preds = pd.read_csv(fname)
        used_file = fname
        print(f"Намерен файл с вашия стек: {used_file}")
        break

if old_preds is None:
    print("⚠️ Не е намерен файл с вашия стек. Моля, посочете ръчно.")
    # Ръчно въвеждане
    manual_file = input("Въведете път към CSV файла с вашите прогнози: ")
    if os.path.exists(manual_file):
        old_preds = pd.read_csv(manual_file)
        used_file = manual_file
    else:
        print("Файлът не съществува. Пропускане на ансамбъла.")
        old_preds = None

# 3. Създаване на ансамбъл (ако имаме двата модела)
if old_preds is not None:
    print(f"\nСъздаване на ансамбъл между {used_file} и H2O...")

    # Проверка за съвпадение на Id-тата
    if not all(old_preds['Id'] == h2o_preds['Id']):
        print("⚠️ Id-тата не съвпадат! Ансамбълът може да е грешен.")

    # Претеглено средно (50/50)
    ensemble_preds = 0.5 * old_preds['SalePrice'] + 0.5 * h2o_preds['SalePrice']

    # Запазване
    ensemble_sub = pd.DataFrame({
        'Id': old_preds['Id'],
        'SalePrice': ensemble_preds
    })
    ensemble_sub.to_csv('submission_ensemble_h2o_v06.csv', index=False)
    print("✅ Ансамбълът е запазен като: submission_ensemble_h2o_v06.csv")

    # Статистика
    print(f"\n📊 Статистика на ансамбъла:")
    print(f"   Min: ${ensemble_preds.min():,.0f}")
    print(f"   Max: ${ensemble_preds.max():,.0f}")
    print(f"   Mean: ${ensemble_preds.mean():,.0f}")
else:
    print("⚠️ Ансамбълът не беше създаден.")

# 4. Допълнително: сравнение на RMSLE (ако имате валидационни данни)
print("\n" + "=" * 80)
print("СРАВНЕНИЕ НА МОДЕЛИТЕ")
print("=" * 80)
print(f"   Вашият стек:         използвайте Kaggle публичен резултат")
print(f"   H2O-3 AutoML:        използвайте Kaggle публичен резултат")
print(f"   Ансамбъл (50/50):    качете submission_ensemble_h2o_v06.csv в Kaggle")
print("=" * 80)
print("\n💡 Препоръка: Качете ансамбъла в Kaggle и вижте кой дава най-добър публичен резултат.")
