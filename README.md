# Análisis Financiero - Aplicación Desktop

Aplicación de escritorio completa para análisis financiero de acciones con Machine Learning.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)
![ML](https://img.shields.io/badge/ML-TensorFlow-orange.svg)
![Pandas](https://img.shields.io/badge/pandas-2.2+-150458.svg)

## 🚀 Características

- **📈 Gráficos de Precios**: Visualización histórica con matplotlib
- **📊 Ratios Financieros**: Análisis completo de métricas empresariales
- **🤖 Predicción ML**: 4 modelos (KNN, Random Forest, LSTM, Neural Network)
- **🏆 Ranking de Portfolio**: Magic Formula de Joel Greenblatt
- **💾 Exportar CSV**: Descarga de datos históricos
- **⚡ Optimizado**: Python 3.12+ con type hints y mejores prácticas

## 📋 Requisitos

- Python 3.12+ (compatible con 3.8+)
- Tkinter (incluido con Python)
- Conexión a internet

## ⚡ Instalación y Uso

### 1. Clonar repositorio
```bash
git clone <url-del-repositorio>
cd ANALISIS_FINANCIERO
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar aplicación
```bash
python app_tkinter.py
```

## 📖 Guía Rápida

### Análisis de Precios
1. Ingresar ticker (ej: `MSFT`, `AAPL`, `GOOGL`)
2. Seleccionar período (1 mes a histórico completo)
3. Click en "Cargar Datos"

### Ratios Financieros
1. Cargar datos de una acción
2. Ir a pestaña "Ratios Financieros"
3. Ver métricas: P/E, ROE, ROA, Current Ratio, etc.
4. Obtener análisis de salud financiera

### Predicción con ML
1. Cargar datos históricos
2. Ir a pestaña "Predicción ML"
3. Seleccionar modelo y Look Back
4. Entrenar y ver métricas (MAE, RMSE)

**Modelos disponibles:**
- **KNN**: Rápido, bueno para análisis inmediato
- **Random Forest**: Balance precisión/velocidad
- **LSTM**: Mejor para series temporales
- **Neural Network**: Flexible y potente

### Ranking de Portfolio
1. Ir a pestaña "Ranking Portfolio"
2. Ingresar tickers separados por coma
3. Ver ranking basado en Magic Formula (P/E + ROA)

## 📊 Ejemplos de Tickers

**Tecnología:** `AAPL`, `MSFT`, `GOOGL`, `META`, `NVDA`, `TSLA`
**Finanzas:** `JPM`, `V`, `BAC`, `GS`
**Consumo:** `WMT`, `KO`, `PG`, `NKE`

## 🗂️ Estructura del Proyecto

```
ANALISIS_FINANCIERO/
├── app_tkinter.py      # Aplicación principal ⭐
├── models.py           # Modelos de Machine Learning
├── requirements.txt    # Dependencias
└── README.md          # Este archivo
```

## ⚙️ Ratios Financieros Analizados

| Categoría | Métricas |
|-----------|----------|
| **Valoración** | P/E Ratio, Forward P/E, P/B, P/S, PEG |
| **Rentabilidad** | Profit Margin, Operating Margin, ROA, ROE |
| **Liquidez** | Current Ratio, Quick Ratio |
| **Deuda** | Debt to Equity, Total Debt |
| **Dividendos** | Dividend Yield, Payout Ratio |

## 🎯 Magic Formula

El ranking combina dos factores clave:
- **P/E Ratio**: Menor es mejor (valoración atractiva)
- **ROA**: Mayor es mejor (rentabilidad)

**Ranking = PE Rank + ROA Rank** (menor es mejor)

## ⏱️ Tiempos Estimados

| Operación | Tiempo |
|-----------|--------|
| Cargar datos | 2-5 seg |
| Ratios financieros | 1-3 seg |
| KNN | 5-10 seg |
| Random Forest | 15-30 seg |
| LSTM | 3-5 min |
| Neural Network | 3-5 min |

## 🐛 Solución de Problemas

### Error: "No se encontraron datos"
- Verificar conexión a internet
- Confirmar ticker válido
- Intentar con otro período

### Error instalando TensorFlow
```bash
pip install --upgrade pip
pip install tensorflow==2.15.0
```

### Aplicación lenta en predicción
- Usar KNN o Random Forest para análisis rápido
- LSTM y Neural Network tardan más (entrenan redes neuronales)

## 📝 Notas Importantes

- Los datos provienen de Yahoo Finance
- Predicciones son **educativas**, no asesoramiento financiero
- Los modelos se entrenan localmente (privacidad garantizada)
- Modelos LSTM/NN requieren TensorFlow instalado

## 🛠️ Tecnologías

- **Python 3.12+**: Optimizado con type hints y mejoras modernas
- **Tkinter**: Interfaz gráfica nativa
- **yfinance 0.2.36+**: Datos financieros de Yahoo Finance
- **pandas 2.2+**: Manipulación de datos de alto rendimiento
- **numpy 1.26+**: Cálculos numéricos
- **matplotlib 3.8+**: Visualización de gráficos
- **scikit-learn 1.4+**: Modelos KNN y Random Forest
- **TensorFlow 2.16+**: Modelos LSTM y Neural Network

## ⚙️ Compatibilidad

| Python Version | Status | Rendimiento |
|----------------|--------|-------------|
| 3.13 | ✅ Soportado | Excelente |
| 3.12 | ✅ Recomendado | Excelente |
| 3.11 | ✅ Soportado | Muy Bueno |
| 3.10 | ✅ Soportado | Bueno |
| 3.9 | ✅ Soportado | Bueno |
| 3.8 | ✅ Mínimo | Aceptable |

**Nota:** Se recomienda Python 3.12+ para mejor rendimiento (10-20% más rápido)

## 📄 Licencia

MIT License - Uso educativo y personal

## 👤 Autor

Desarrollado para análisis financiero educativo

---

**⭐ Si te resulta útil, dale una estrella al repositorio!**
