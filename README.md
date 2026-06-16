# House Prices - Advanced Regression Techniques

Финален проект по Machine Learning.

**Цел:** Прогнозиране на цени на жилища, класификация на скъпи/евтини имоти, клъстериране на квартали, намаляване на размерността и MLflow.

**Постигнат резултат:** RMSLE **0.11974** (362-ро място от 5115 участници в Kaggle).

**Модели:** Linear Regression, Ridge, Lasso, ElasticNet, RANSAC, Random Forest, XGBoost, LightGBM, Stacking (5 модела + meta-learner).

**Техники:** Feature engineering (TotalSF, HouseAge, TopNeighborhood, Qual_Area и др.), кръстосано валидиране, GridSearchCV, MLflow tracking, **последователно намаляване на регуларизацията на мета-модела**.

## 📈 Прогресия на резултатите

| Версия | Meta alpha | Public RMSLE | Δ от предишната |
|--------|------------|--------------|-----------------|
| Baseline | 0.5 | 0.11988 | – |
| v_01 | 0.4 | 0.11986 | -0.00002 |
| v_02 | 0.3 | 0.11983 | -0.00003 |
| v_03 | 0.2 | 0.11981 | -0.00002 |
| v_04 | 0.1 | 0.11977 | -0.00004 |
| v_05 | 0.05 | 0.11975 | -0.00002 |
| **v_06** | **0.02** | **0.11974** | **-0.00001** |

> Включени са само успешните версии. Версии, които не подобряват резултата (напр. alpha=0.01, експерименти с CatBoost/Lasso), не са част от хранилището.

## 🧠 Ключов подход

Основното подобрение идва от **систематичното намаляване на регуларизацията (alpha)** на Ridge мета-модела – от 0.5 до 0.02 – което последователно намалява RMSLE.

Коефициентите на мета-модела при alpha=0.02 са:
- Random Forest: -0.462
- XGBoost: 0.145
- LightGBM: 0.704
- Gradient Boosting: 0.591
- Ridge (като базов модел): 0.013 (почти нулев принос)

**Структура:**
- `house_prices_model.py` – базов Random Forest
- `house_prices_xgboost.py` – XGBoost
- `house_prices_lightgbm.py` – LightGBM
- `house_prices_stacking.py` – Stacking (3 модела)
- `house_prices_improved.py` – Базов Stacking (5 модела) – RMSLE 0.11988
- `house_prices_improved_v_01.py` – alpha=0.4 – RMSLE 0.11986
- `house_prices_improved_v_02.py` – alpha=0.3 – RMSLE 0.11983
- `house_prices_improved_v_03.py` – alpha=0.2 – RMSLE 0.11981
- `house_prices_improved_v_04.py` – alpha=0.1 – RMSLE 0.11977
- `house_prices_improved_v_05.py` – alpha=0.05 – RMSLE 0.11975
- `house_prices_improved_v_06.py` – **Финален модел** – alpha=0.02 – **RMSLE 0.11974**
- `submissions/` – CSV файлове с прогнози

**Възпроизвеждане:**  
`pip install -r requirements.txt`  
`python house_prices_improved_v_06.py`

**Автор:** Todor Yankov  
**Дата:** 16 Юни 2026