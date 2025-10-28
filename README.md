# Análisis Financiero - Aplicación Desktop

Aplicación de escritorio para análisis financiero de acciones con Machine Learning.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)
![ML](https://img.shields.io/badge/ML-TensorFlow-orange.svg)
![Pandas](https://img.shields.io/badge/pandas-2.2+-150458.svg)

## Características

- Gráficos de precios: visualización histórica con matplotlib
- Ratios financieros: métricas de valoración, rentabilidad, liquidez y deuda
- Predicción ML: KNN, Random Forest, LSTM y Red Neuronal Densa
- Ranking de portfolio: Magic Formula (P/E + ROA)
- Exportar CSV de datos históricos
- Optimizado para Python 3.12+

## Requisitos

- Python 3.12+ (compatible con 3.8+)
- Tkinter (incluido con Python)
- Conexión a internet

## Instalación y uso

### 1. Clonar repositorio
```
git clone <url-del-repositorio>
cd ANALISIS_FINANCIERO
```

### 2. Instalar dependencias
```
pip install -r requirements.txt
```

### 3. Ejecutar aplicación
```
python app_tkinter.py
```

## Guía rápida

### Análisis de precios
1. Ingresar ticker (ej: `MSFT`, `AAPL`, `GOOGL`)
2. Seleccionar período (1 mes a histórico completo)
3. Click en "Cargar Datos"

### Ratios financieros
1. Cargar datos de una acción
2. Ir a pestaña "Ratios Financieros"
3. Ver métricas: P/E, ROE, ROA, Current Ratio, etc.
4. Obtener análisis de salud financiera

### Predicción con ML
1. Cargar datos históricos
2. Ir a pestaña "Predicción ML"
3. Seleccionar modelo y Look Back
4. Entrenar y ver métricas (MAE, RMSE)

### Ranking de portfolio
1. Ir a pestaña "Ranking Portfolio"
2. Ingresar tickers separados por coma
3. Ver ranking basado en Magic Formula (P/E + ROA)

## Ejemplos de tickers

**Tecnología:** `AAPL`, `MSFT`, `GOOGL`, `META`, `NVDA`, `TSLA`
**Finanzas:** `JPM`, `V`, `BAC`, `GS`
**Consumo:** `WMT`, `KO`, `PG`, `NKE`

## Estructura del proyecto

```
ANALISIS_FINANCIERO/
├── app_tkinter.py      # Aplicación principal
├── models.py           # Modelos de Machine Learning
├── requirements.txt    # Dependencias
├── README.md           # Este archivo
└── models_saved/       # Carpeta para modelos guardados
```

## Ratios financieros analizados

| Categoría      | Métricas                                      |
|----------------|-----------------------------------------------|
| Valoración     | P/E, Forward P/E, P/B, P/S, PEG               |
| Rentabilidad   | Profit Margin, Operating Margin, ROA, ROE     |
| Liquidez       | Current Ratio, Quick Ratio                    |
| Deuda          | Debt to Equity, Total Debt                    |
| Dividendos     | Dividend Yield, Payout Ratio                  |

## Magic Formula

Ranking combinado de:
- P/E Ratio (menor es mejor)
- ROA (mayor es mejor)

Ranking = PE Rank + ROA Rank (menor es mejor)

## Tiempos estimados

| Operación        | Tiempo        |
|------------------|---------------|
| Cargar datos     | 2-5 seg       |
| Ratios           | 1-3 seg       |
| KNN              | 5-10 seg      |
| Random Forest    | 15-30 seg     |
| LSTM             | 3-5 min       |
| Neural Network   | 3-5 min       |

## Solución de problemas

### "No se encontraron datos"
- Verificar conexión a internet
- Confirmar ticker válido
- Probar con otro período

### Error instalando TensorFlow
```
pip install --upgrade pip
pip install tensorflow==2.15.0
```

### Predicción lenta
- Usar KNN o Random Forest para análisis rápido
- LSTM y NN tardan más (entrenan redes neuronales)

## Notas importantes

- Los datos provienen de Yahoo Finance (yfinance)
- Predicciones con fines educativos; no es asesoramiento financiero
- Los modelos se entrenan localmente (privacidad)
- Modelos LSTM/NN requieren TensorFlow instalado
- Para datos fundamentales se usa `fast_info` cuando es posible; `info` se usa como respaldo si falla o faltan campos

## Tecnologías

- Python 3.12+
- Tkinter
- yfinance 0.2.36+
- pandas 2.2+
- numpy 1.26+
- matplotlib 3.8+
- scikit-learn 1.4+
- TensorFlow 2.16+

## Compatibilidad

| Python | Estado       | Rendimiento |
|--------|--------------|-------------|
| 3.13   | Soportado    | Excelente   |
| 3.12   | Recomendado  | Excelente   |
| 3.11   | Soportado    | Muy bueno   |
| 3.10   | Soportado    | Bueno       |
| 3.9    | Soportado    | Bueno       |
| 3.8    | Mínimo       | Aceptable   |

## Licencia

MIT License - Uso educativo y personal

## Autor

Desarrollado para análisis financiero educativo

---

Si te resultó útil, dale una estrella al repositorio.
