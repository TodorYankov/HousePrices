# House Prices - Advanced Regression Techniques

Финален проект по Machine Learning.

**Цел:** Прогнозиране на цени на жилища, класификация на скъпи/евтини имоти, клъстериране на квартали, намаляване на размерността и MLflow.

**🏆 Финален резултат:** RMSLE **0.11940**, класиране **302 / 5094** (топ 6%) в Kaggle.

## 📈 Прогресия на резултатите (публичен Kaggle RMSLE)

| Версия | Мета-модел | Alpha | Базови модели | RMSLE | Δ |
|--------|------------|-------|---------------|-------|---|
| Baseline (improved) | Ridge | 0.5 | 5 (RF, XGB, LGB, GB, Ridge) | 0.11988 | – |
| v_01 | Ridge | 0.4 | 5 | 0.11986 | -0.00002 |
| v_02 | Ridge | 0.3 | 5 | 0.11983 | -0.00003 |
| v_03 | Ridge | 0.2 | 5 | 0.11981 | -0.00002 |
| v_04 | Ridge | 0.1 | 5 | 0.11977 | -0.00004 |
| v_05 | Ridge | 0.05 | 5 | 0.11975 | -0.00002 |
| v_06 | Ridge | 0.02 | 5 | 0.11974 | -0.00001 |
| v_07 | Ridge | 0.01 | 5 | 0.11974 | 0 |
| **Ensemble (v_07 + H2O)** | – | – | **5 + H2O AutoML** | **0.11940** | **-0.00034** |

> **Най-добър резултат:** Ансамбъл 50/50 между ръчен стек (v_07) и H2O-3 AutoML.

## 🧠 Ключов подход

Основното подобрение идва от **систематичното намаляване на регуларизацията (alpha)** на Ridge мета-модела – от 0.5 до 0.02 – което последователно намалява RMSLE.

Коефициентите на мета-модела при alpha=0.02 (v_06) са:
- Random Forest: -0.462
- XGBoost: 0.145
- LightGBM: 0.704
- Gradient Boosting: 0.591
- Ridge (като базов модел): 0.013 (почти нулев принос)

### H2O-3 AutoML експеримент

Като допълнителен експеримент тествах H2O-3 AutoML (50 модела, 32 минути). Най-добрият му модел (StackedEnsemble) постигна CV RMSLE 0.12609. Комбинирах го с v_07 в 50/50 ансамбъл, което доведе до финалния резултат **0.11940** и класиране **302/5094**.

## 📂 Структура на хранилището

- `house_prices_analysis.ipynb` – основен анализ и експерименти
- `house_prices_model.py` – базов Random Forest
- `house_prices_xgboost.py` – XGBoost
- `house_prices_lightgbm.py` – LightGBM
- `house_prices_stacking.py` – Stacking (3 модела)
- `house_prices_improved.py` – Базов Stacking (5 модела) – RMSLE 0.11988
- `house_prices_improved_v_01.py` до `v_06.py` – последователни версии с намаляваща alpha
- `house_prices_improved_v_06.py` – Финален стекинг модел (alpha=0.02)
- `house_prices_h2o_autoML.py` – H2O-3 AutoML скрипт (15-минутен тест)
- `house_prices_h2o_full.py` – H2O-3 AutoML пълен експеримент (2 часа)
- `house_prices_h2o_ensemble.py` – създава ансамбъл v_07 + H2O
- `submission_ensemble_h2o_v06.csv` – **най-доброто submission (RMSLE 0.11940)**
- `submissions/` – други CSV файлове с прогнози
- `README.md` – този файл

## 🚀 Възпроизвеждане

### Бързо възпроизвеждане (най-добър самостоятелен модел)

```bash
pip install -r requirements.txt
python house_prices_improved_v_06.py
Това ще генерира submission_house_prices_improved_v_06.csv с RMSLE 0.11974.

Възпроизвеждане на най-добрия ансамбъл (RMSLE 0.11940)
За да възпроизведете финалния ансамбъл (v_07 + H2O-3 AutoML), изпълнете следните стъпки:

Стартирайте H2O-3 AutoML (отнема ~32 минути):

bash
python house_prices_h2o_full.py
Това ще генерира submission_h2o_autoML_full.csv.

Създайте ансамбъла:

bash
python house_prices_h2o_ensemble.py
Това ще генерира submission_ensemble_h2o_v06.csv с RMSLE 0.11940 (най-добър резултат).

Изисквания за ансамбъла: H2O-3 и Java (виж requirements.txt).

📊 MLflow
Запазени са всички експерименти – може да се разгледат с:

bash
mlflow ui
👤 Автор
Todor Yankov
Дата: 18 Юни 2026