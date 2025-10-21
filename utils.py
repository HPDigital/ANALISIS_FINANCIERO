"""
Utilidades y funciones auxiliares para análisis financiero
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from datetime import datetime


def download_stock_data(symbol, start_date, end_date):
    """
    Descarga datos históricos de una acción desde Yahoo Finance
    
    Args:
        symbol (str): Símbolo de la acción (ej: 'MSFT')
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'
        
    Returns:
        pd.DataFrame: DataFrame con datos históricos de la acción
    """
    print(f"Descargando datos de {symbol}...")
    data = yf.download(symbol, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data.dropna(inplace=True)
    print(f"✓ Datos descargados: {len(data)} registros")
    return data


def calculate_moving_averages(data, windows=[50, 100, 200]):
    """
    Calcula medias móviles para diferentes ventanas de tiempo
    
    Args:
        data (pd.DataFrame): DataFrame con columna 'Close'
        windows (list): Lista de ventanas para calcular MA
        
    Returns:
        pd.DataFrame: DataFrame con columnas MA añadidas
    """
    data = data.copy()
    for window in windows:
        data[f'MA{window}'] = data['Close'].rolling(window).mean()
    return data


def create_dataset(data, look_back=100):
    """
    Crea dataset para predicción de series temporales
    
    Args:
        data (np.array): Array de datos escalados
        look_back (int): Número de pasos hacia atrás para usar como features
        
    Returns:
        tuple: (X, Y) arrays para entrenamiento
    """
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)


def split_train_test(data, train_ratio=0.8):
    """
    Divide datos en conjuntos de entrenamiento y prueba
    
    Args:
        data (pd.DataFrame): DataFrame completo
        train_ratio (float): Proporción de datos para entrenamiento
        
    Returns:
        tuple: (train_data, test_data)
    """
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"Datos de entrenamiento: {len(train_data)} registros")
    print(f"Datos de prueba: {len(test_data)} registros")
    
    return train_data, test_data


def scale_data(train_data, test_data, feature_column='Close'):
    """
    Escala datos usando MinMaxScaler
    
    Args:
        train_data (pd.DataFrame): Datos de entrenamiento
        test_data (pd.DataFrame): Datos de prueba
        feature_column (str): Nombre de la columna a escalar
        
    Returns:
        tuple: (train_scaled, test_scaled, scaler)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data[[feature_column]])
    test_scaled = scaler.transform(test_data[[feature_column]])
    
    return train_scaled, test_scaled, scaler


def plot_price_vs_ma(data, stock_symbol, ma_windows=[100]):
    """
    Grafica precio vs medias móviles
    
    Args:
        data (pd.DataFrame): DataFrame con precios y MA
        stock_symbol (str): Símbolo de la acción
        ma_windows (list): Ventanas de MA a graficar
    """
    fig = go.Figure()
    
    # Precio de cierre
    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['Close'], 
        mode='lines', 
        name='Precio de Cierre',
        line=dict(color='blue', width=2)
    ))
    
    # Medias móviles
    colors = ['orange', 'red', 'green']
    for i, window in enumerate(ma_windows):
        ma_col = f'MA{window}'
        if ma_col in data.columns:
            fig.add_trace(go.Scatter(
                x=data['Date'], 
                y=data[ma_col], 
                mode='lines', 
                name=f'MA{window}',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    fig.update_layout(
        title=f'{stock_symbol} - Precio vs Medias Móviles',
        xaxis_title='Fecha',
        yaxis_title='Precio (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    # Guardar en archivo en lugar de mostrar
    filename = f'graficos/{stock_symbol}_precio_vs_ma.html'
    fig.write_html(filename)
    print(f"✓ Gráfico guardado en: {filename}")


def plot_predictions(test_data, y_test, y_pred, model_name, look_back=100):
    """
    Grafica precios reales vs predicciones
    
    Args:
        test_data (pd.DataFrame): Datos de prueba con fechas
        y_test (np.array): Valores reales
        y_pred (np.array): Valores predichos
        model_name (str): Nombre del modelo
        look_back (int): Ventana de look_back usada
    """
    # Padding para alinear con fechas
    y_test_padded = np.concatenate((np.zeros((look_back, 1)), y_test))
    
    fig = go.Figure()
    
    # Precios reales
    fig.add_trace(go.Scatter(
        x=test_data['Date'], 
        y=y_test_padded.flatten(), 
        mode='lines', 
        name='Precio Real',
        line=dict(color='blue', width=2)
    ))
    
    # Predicciones
    fig.add_trace(go.Scatter(
        x=test_data['Date'].iloc[look_back:], 
        y=y_pred.flatten(), 
        mode='lines', 
        name='Predicción',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{model_name} - Precio Real vs Predicción',
        xaxis_title='Fecha',
        yaxis_title='Precio (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    # Guardar en archivo en lugar de mostrar
    filename = f'graficos/{model_name.replace(" ", "_")}_prediccion.html'
    fig.write_html(filename)
    print(f"✓ Gráfico guardado en: {filename}")


def print_metrics(mae, mse, rmse, model_name):
    """
    Imprime métricas de evaluación del modelo
    
    Args:
        mae (float): Mean Absolute Error
        mse (float): Mean Squared Error
        rmse (float): Root Mean Squared Error
        model_name (str): Nombre del modelo
    """
    print("\n" + "="*60)
    print(f"📊 MÉTRICAS DE EVALUACIÓN - {model_name}")
    print("="*60)
    print(f"Mean Absolute Error (MAE):  ${mae:.2f}")
    print(f"Mean Squared Error (MSE):   ${mse:.2f}")
    print(f"Root Mean Squared Error:    ${rmse:.2f}")
    print("="*60 + "\n")


def get_stock_info(symbol):
    """
    Obtiene información general de una acción
    
    Args:
        symbol (str): Símbolo de la acción
        
    Returns:
        dict: Diccionario con información de la empresa
    """
    ticker = yf.Ticker(symbol)
    return ticker.info


def create_portfolio_dataframe(stock_list):
    """
    Crea DataFrame con información de múltiples acciones
    
    Args:
        stock_list (list): Lista de símbolos de acciones
        
    Returns:
        pd.DataFrame: DataFrame con información de todas las acciones
    """
    df_list = []
    
    for ticker in stock_list:
        try:
            print(f"Obteniendo datos de {ticker}...")
            info = yf.Ticker(ticker).info
            df_list.append(pd.DataFrame([info]))
        except Exception as e:
            print(f"⚠ Error con {ticker}: {str(e)}")
            continue
    
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()



if __name__ == "__main__":
    pass
