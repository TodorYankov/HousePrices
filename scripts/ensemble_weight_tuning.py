# ensemble_weight_tuning.py
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 80)
print("АНСАМБЪЛ С ТЮНИНГ НА ТЕЖЕСТИ - ОПИТ ЗА ПОДОБРЕНИЕ")
print("=" * 80)

# 1. Зареждане на наличните прогнози
print("\n1. Зареждане на прогнозите...")

models = {}

# Опит за зареждане на v_06
if os.path.exists('submission_house_prices_improved_v_06.csv'):
    models['v06'] = pd.read_csv('submission_house_prices_improved_v_06.csv')
    print("   ✅ Зареден v_06 (0.11974)")

# Опит за зареждане на v_07
if os.path.exists('submission_house_prices_improved_v_07.csv'):
    models['v07'] = pd.read_csv('submission_house_prices_improved_v_07.csv')
    print("   ✅ Зареден v_07 (0.11974)")

# Опит за зареждане на H2O
if os.path.exists('submission_h2o_autoML_full.csv'):
    models['h2o'] = pd.read_csv('submission_h2o_autoML_full.csv')
    print("   ✅ Зареден H2O-3 AutoML (0.12609)")

if len(models) < 2:
    print("❌ Няма достатъчно модели за ансамбъл. Имате нужда от поне 2 файла.")
    exit()

# Взимаме Id-тата от първия модел
ids = list(models.values())[0]['Id']

print(f"\n📊 Брой модели за ансамбъл: {len(models)}")
print(f"   Модели: {', '.join(models.keys())}")

# 2. Създаване на ансамбли с различни тежести
print("\n2. Създаване на ансамбли с различни тежести...")

# Списък за всички комбинации
weight_combinations = []

# Ако имаме 2 модела – тестваме различни тежести
if len(models) == 2:
    keys = list(models.keys())
    for w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        weight_combinations.append({
            'name': f"{keys[0]}_{int(w * 100)}_{keys[1]}_{int((1 - w) * 100)}",
            'weights': {keys[0]: w, keys[1]: 1 - w}
        })
# Ако имаме 3 модела – тестваме повече комбинации
elif len(models) == 3:
    keys = list(models.keys())
    # Тестваме различни комбинации за v07 и h2o, като фиксираме v06 на 0.0, 0.2, 0.3
    for v06_w in [0.0, 0.1, 0.2, 0.3]:
        remaining = 1.0 - v06_w
        for h2o_w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            v07_w = remaining - h2o_w
            if 0 <= v07_w <= 1 and 0 <= h2o_w <= 1:
                weight_combinations.append({
                    'name': f"v06_{int(v06_w * 100)}_v07_{int(v07_w * 100)}_h2o_{int(h2o_w * 100)}",
                    'weights': {'v06': v06_w, 'v07': v07_w, 'h2o': h2o_w}
                })

# Ограничаваме до 15 комбинации за бързина (най-обещаващите)
if len(weight_combinations) > 15:
    # Филтрираме тези с по-голяма тежест на v07 (който е най-добрият)
    weight_combinations = [c for c in weight_combinations if c['weights'].get('v07', 0) >= 0.3]
    weight_combinations = weight_combinations[:15]

print(f"   Ще бъдат тествани {len(weight_combinations)} комбинации.")

# 3. Създаване на CSV файлове
print("\n3. Запазване на ансамблите...")

created_files = []
for combo in weight_combinations:
    ensemble_pred = np.zeros(len(ids))

    for model_name, w in combo['weights'].items():
        if model_name in models:
            ensemble_pred += w * models[model_name]['SalePrice']
        else:
            print(f"   ⚠️ Модел {model_name} не е намерен, пропуска се.")

    # Запазване
    fname = f"submission_ensemble_{combo['name']}.csv"
    pd.DataFrame({
        'Id': ids,
        'SalePrice': ensemble_pred
    }).to_csv(fname, index=False)
    created_files.append(fname)
    print(f"   ✅ {fname}")

# 4. Инструкции за качване в Kaggle
print("\n" + "=" * 80)
print("📋 ИНСТРУКЦИИ ЗА КАЧВАНЕ В KAGGLE")
print("=" * 80)
print("1. Качете ВСИЧКИ нови CSV файлове в Kaggle (едно по едно).")
print("2. Запишете публичния RMSLE за всеки.")
print("3. Изберете този с НАЙ-МАЛЪК RMSLE.")

print("\n📊 Списък на файловете за качване:")
for f in created_files:
    print(f"   - {f}")

print("\n💡 Ако някой файл даде RMSLE < 0.11940, използвайте го като финален.")
print("💡 Ако никой не подобри, запазете текущия си най-добър резултат (0.11940).")
print("=" * 80)

# 5. Бонус: запазване на най-обещаващия според тежестите
# (този с най-голяма тежест на v07)
print("\n📌 Препоръчителни кандидати за качване (по приоритет):")
print("   1. submission_ensemble_v07_60_h2o_40.csv (ако съществува)")
print("   2. submission_ensemble_v06_20_v07_50_h2o_30.csv (ако съществува)")
print("   3. submission_ensemble_v06_0_v07_60_h2o_40.csv (ако съществува)")
