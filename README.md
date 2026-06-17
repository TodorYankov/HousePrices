# House Prices - Advanced Regression Techniques

Финален проект по Machine Learning.

**Цел:** Прогнозиране на цени на жилища, класификация на скъпи/евтини имоти, клъстериране на квартали, намаляване на размерността и MLflow.

**Постигнат резултат:** RMSLE **0.11974** (362-ро място от 5115 участници в Kaggle).

**Модели:** Linear Regression, Ridge, Lasso, ElasticNet, RANSAC, Random Forest, XGBoost, LightGBM, Gradient Boosting, CatBoost, ExtraTrees, Stacking (5-7 модела + meta-learner).

**Техники:** Feature engineering (TotalSF, HouseAge, TopNeighborhood, Qual_Area и др.), кръстосано валидиране, GridSearchCV, Optuna, MLflow tracking, **последователно намаляване на регуларизацията на мета-модела**, hold-out валидация за избор на мета-модел.

## 📈 Прогресия на резултатите (публичен Kaggle RMSLE)

| Версия | Мета-модел | Alpha | Базови модели | RMSLE | Δ |
|--------|------------|-------|---------------|-------|---|
| Baseline (improved) | Ridge | 0.5 | 5 (RF, XGB, LGB, GB, Ridge) | 0.11988 | – |
| v_01 | Ridge | 0.4 | 5 | 0.11986 | -0.00002 |
| v_02 | Ridge | 0.3 | 5 | 0.11983 | -0.00003 |
| v_03 | Ridge | 0.2 | 5 | 0.11981 | -0.00002 |
| v_04 | Ridge | 0.1 | 5 | 0.11977 | -0.00004 |
| v_05 | Ridge | 0.05 | 5 | 0.11975 | -0.00002 |
| **v_06** | **Ridge** | **0.02** | **5** | **0.11974** | **-0.00001** |
| v_07 | Ridge | 0.01 | 5 | 0.11974 | 0 |
| enhanced (тест) | LightGBM | – | 7 (добавени CatBoost, ExtraTrees) | (не е качван) | – |

> Забележка: v_06 е финалният модел, качен в Kaggle. По-нататъшни експерименти с 7 базови модела и LightGBM мета показаха пренастройване и не бяха качвани.

## 🧠 Ключов подход

Основното подобрение идва от **систематичното намаляване на регуларизацията (alpha)** на Ridge мета-модела – от 0.5 до 0.02 – което последователно намалява RMSLE.

Коефициентите на мета-модела при alpha=0.02 (v_06) са:
- Random Forest: -0.462
- XGBoost: 0.145
- LightGBM: 0.704
- Gradient Boosting: 0.591
- Ridge (като базов модел): 0.013 (почти нулев принос)

### Експерименти с разширен набор от базови модели и мета-модели

За да проверим дали можем да постигнем още по-добър резултат, разработихме тестови скриптове:

- `test_meta_models_enhanced.py` – 7 базови модела (RF, XGB, LGB, GB, Ridge, CatBoost, ExtraTrees), оценява различни мета-модели върху OOF прогнози.
- **Резултат:** LightGBM като мета-модел дава най-нисък RMSLE върху OOF (0.00681), но това е оптимистична оценка.

- `test_meta_models_with_validation.py` – същите 7 базови модела, но мета-моделите се оценяват върху отделен 20% hold-out от OOF данните.
- **Резултат:** LightGBM мета се пренастройва – RMSLE скача от 0.00681 на **0.00992**. Ridge(alpha=0.5) е най-стабилен с **0.00918** hold-out RMSLE.

Това потвърждава, че **Ridge с лека регуларизация е най-добрият избор за мета-модел** за този набор от данни.

## 📂 Структура на хранилището

- `house_prices_model.py` – базов Random Forest
- `house_prices_xgboost.py` – XGBoost
- `house_prices_lightgbm.py` – LightGBM
- `house_prices_stacking.py` – Stacking (3 модела)
- `house_prices_improved.py` – Базов Stacking (5 модела) – RMSLE 0.11988
- `house_prices_improved_v_01.py` до `v_06.py` – последователни версии с намаляваща alpha
- **`house_prices_improved_v_06.py`** – **Финален модел** (alpha=0.02) – **RMSLE 0.11974**
- `test_meta_models_enhanced.py` – тест с 7 базови модела
- `test_meta_models_with_validation.py` – тест с hold-out валидация
- `submissions/` – CSV файлове с прогнози

## 🚀 Възпроизвеждане

```bash
pip install -r requirements.txt
python house_prices_improved_v_06.py
📊 MLflow
Запазени са всички експерименти – може да се разгледат с:

bash
mlflow ui
👤 Автор
Todor Yankov
Дата: 17 Юни 2026
