# House Prices - Advanced Regression Techniques

Финален проект по Machine Learning.

**Цел:** Прогнозиране на цени на жилища, класификация на скъпи/евтини имоти, клъстериране на квартали, намаляване на размерността и MLflow.

**Постигнат резултат:** RMSLE 0.11988 (264-то място от ~5000 участници в Kaggle).

**Модели:** Linear Regression, Ridge, Lasso, ElasticNet, RANSAC, Random Forest, XGBoost, LightGBM, Stacking (5 модела + meta-learner).

**Техники:** Feature engineering (TotalSF, HouseAge, TopNeighborhood, Qual_Area и др.), кръстосано валидиране, GridSearchCV, MLflow tracking.

**Структура:**
- `house_prices_model.py` – базов Random Forest
- `house_prices_xgboost.py` – XGBoost
- `house_prices_lightgbm.py` – LightGBM
- `house_prices_stacking.py` – Stacking (3 модела)
- `house_prices_improved.py` – Финален Stacking (5 модела) – RMSLE 0.11988
- `submissions/` – CSV файлове с прогнози

**Възпроизвеждане:**  
`pip install -r requirements.txt`  
`python house_prices_improved.py`

**Автор:** Todor Yankov  
**Дата:** Май 2026