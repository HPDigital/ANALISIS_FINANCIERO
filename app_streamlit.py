"""
Aplicación Streamlit para Análisis Financiero
Visualización interactiva de datos de acciones en tiempo real
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Análisis Financiero",
    page_icon="📈",
    layout="wide"
)

# Título
st.title("📈 Análisis Financiero Interactivo")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuración")

# Input del ticker
ticker_symbol = st.sidebar.text_input(
    "Símbolo de la Acción:",
    value="MSFT",
    help="Ejemplo: AAPL, GOOGL, MSFT, TSLA"
).upper()

# Rango de fechas
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.sidebar.date_input(
        "Fecha Inicio:",
        value=datetime.now() - timedelta(days=365)
    )
with col2:
    end_date = st.sidebar.date_input(
        "Fecha Fin:",
        value=datetime.now()
    )

# Botón
if st.sidebar.button("📊 Cargar Datos", type="primary"):
    with st.spinner(f"Descargando datos de {ticker_symbol}..."):
        try:
            # Descargar datos directamente
            data = yf.download(
                ticker_symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if data.empty or len(data) == 0:
                st.error(f"❌ No se encontraron datos para {ticker_symbol}")
                st.info("💡 Verifica el símbolo. Ejemplos: AAPL, MSFT, GOOGL, TSLA, NVDA")
            else:
                # Preparar datos
                data = data.reset_index()
                st.success(f"✅ {len(data)} registros cargados exitosamente")
                
                # Información general
                try:
                    ticker_obj = yf.Ticker(ticker_symbol)
                    info = ticker_obj.info
                    
                    st.subheader(f"📊 {ticker_symbol} - {info.get('shortName', ticker_symbol)}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Precio Actual", f"${info.get('currentPrice', info.get('regularMarketPrice', 0)):.2f}")
                    col2.metric("Cambio %", f"{info.get('regularMarketChangePercent', 0):.2f}%")
                    col3.metric("Volumen", f"{info.get('volume', info.get('regularMarketVolume', 0)):,}")
                    col4.metric("Cap. Mercado", f"${info.get('marketCap', 0)/1e9:.2f}B")
                except:
                    st.subheader(f"📊 {ticker_symbol}")
                
                st.markdown("---")
                
                # Gráfico de velas
                st.subheader("📈 Gráfico de Precios")
                
                fig = go.Figure(data=[go.Candlestick(
                    x=data['Date'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close']
                )])
                
                fig.update_layout(
                    title=f"{ticker_symbol} - Evolución del Precio",
                    xaxis_title="Fecha",
                    yaxis_title="Precio (USD)",
                    height=500,
                    xaxis_rangeslider_visible=False,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico de volumen
                st.subheader("📊 Volumen de Transacciones")
                
                fig2 = go.Figure(data=[go.Bar(
                    x=data['Date'],
                    y=data['Volume'],
                    marker_color='lightblue'
                )])
                
                fig2.update_layout(
                    title=f"{ticker_symbol} - Volumen",
                    xaxis_title="Fecha",
                    yaxis_title="Volumen",
                    height=300,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Tabla de datos
                st.subheader("📋 Datos Históricos (Últimos 50 registros)")
                
                display_data = data.tail(50).copy()
                display_data = display_data.sort_values('Date', ascending=False)
                
                st.dataframe(
                    display_data,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Estadísticas
                st.subheader("📊 Estadísticas Descriptivas")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Precio de Cierre:**")
                    st.write(data['Close'].describe())
                
                with col2:
                    st.write("**Volumen:**")
                    st.write(data['Volume'].describe())
                
                # Descargar
                st.markdown("---")
                csv = data.to_csv(index=False)
                st.download_button(
                    label="💾 Descargar Datos CSV",
                    data=csv,
                    file_name=f"{ticker_symbol}_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error al cargar datos: {str(e)}")
            st.info("💡 Soluciones posibles:")
            st.markdown("""
            - Verifica tu conexión a internet
            - Confirma que el símbolo sea correcto
            - Intenta con otro símbolo (ej: MSFT, AAPL, GOOGL)
            - Cambia el rango de fechas
            """)

else:
    # Pantalla inicial
    st.info("👈 Configura los parámetros en la barra lateral y presiona 'Cargar Datos'")
    
    st.markdown("""
    ### 🎯 Características:
    - 📊 Datos históricos de acciones en tiempo real
    - 📈 Gráficos interactivos de velas japonesas
    - 📉 Análisis de volumen de transacciones
    - 📋 Tabla de datos históricos
    - 📊 Estadísticas descriptivas
    - 💾 Descarga de datos en formato CSV
    
    ### 💡 Tickers Populares:
    
    **Tecnología:**
    - **AAPL** - Apple Inc.
    - **MSFT** - Microsoft Corporation
    - **GOOGL** - Alphabet Inc. (Google)
    - **META** - Meta Platforms (Facebook)
    - **NVDA** - NVIDIA Corporation
    - **TSLA** - Tesla Inc.
    - **AMZN** - Amazon.com Inc.
    
    **Finanzas:**
    - **JPM** - JPMorgan Chase & Co.
    - **V** - Visa Inc.
    - **MA** - Mastercard Inc.
    
    **Consumo:**
    - **WMT** - Walmart Inc.
    - **KO** - The Coca-Cola Company
    - **PG** - Procter & Gamble
    
    ### 📝 Instrucciones:
    1. Ingresa el símbolo de la acción (ticker)
    2. Selecciona el rango de fechas
    3. Presiona el botón "📊 Cargar Datos"
    4. Explora los gráficos y datos interactivos
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'><p>💹 Datos proporcionados por Yahoo Finance</p></div>",
    unsafe_allow_html=True
)
