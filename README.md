# Análisis Financiero

Predicción de precios de acciones con Machine Learning, ranking de portfolio y análisis de ratios financieros.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Ejecutar todo el análisis
python run_all.py

# O ejecutar scripts individuales:
python stock_prediction.py      # Predicción con ML (KNN, Random Forest, LSTM, NN)
python portfolio_ranking.py     # Ranking de portfolio (Magic Formula)
python financial_ratios.py      # Análisis de ratios financieros

# App web interactiva
streamlit run app_streamlit.py
```

## Estructura

- `stock_prediction.py` - Predicción de precios con 4 modelos ML
- `portfolio_ranking.py` - Ranking de acciones (Magic Formula)
- `financial_ratios.py` - Análisis de métricas financieras
- `app_streamlit.py` - Aplicación web interactiva
- `models.py` - Definición de modelos
- `utils.py` - Funciones auxiliares
- `run_all.py` - Ejecuta todos los análisis
