"""
Aplicación Tkinter para Análisis Financiero Completo
Incluye: Análisis de precios, ratios financieros, ranking de portfolio y predicción ML

Optimizado para Python 3.12+
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import threading
from sklearn.preprocessing import MinMaxScaler
from models import KNNModel, RandomForestModel, LSTMModel, NeuralNetworkModel
import os
import warnings

# Suprimir warnings de pandas deprecations
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Constantes
PERIOD_OPTIONS = {
    "1 Mes": "1mo",
    "3 Meses": "3mo",
    "6 Meses": "6mo",
    "1 Año": "1y",
    "2 Años": "2y",
    "5 Años": "5y",
    "Todo": "max"
}


class FinancialAnalysisApp(tk.Tk):
    """Aplicación principal de análisis financiero"""

    def __init__(self) -> None:
        super().__init__()
        self.title("Análisis Financiero - Aplicación Completa")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        # Variables de datos
        self.df_prices: pd.DataFrame = pd.DataFrame()
        self.ticker_info: dict = {}
        self.financial_ratios: dict = {}
        self.predictions: dict = {}

        # Construir UI
        self._build_ui()

    def _build_ui(self):
        """Construye la interfaz de usuario"""
        # Panel de controles superior
        controls_frame = ttk.Frame(self, padding=10)
        controls_frame.pack(side=tk.TOP, fill=tk.X)

        # Ticker
        ttk.Label(controls_frame, text="Ticker:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.ticker_var = tk.StringVar(value="MSFT")
        self.ticker_entry = ttk.Entry(controls_frame, textvariable=self.ticker_var, width=12)
        self.ticker_entry.grid(row=0, column=1, sticky=tk.W)

        # Período
        ttk.Label(controls_frame, text="Período:").grid(row=0, column=2, sticky=tk.W, padx=(15, 5))
        self.period_var = tk.StringVar(value="1 Año")
        self.period_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.period_var,
            values=list(PERIOD_OPTIONS.keys()),
            state="readonly",
            width=12
        )
        self.period_combo.grid(row=0, column=3, sticky=tk.W)

        # Botones
        self.fetch_btn = ttk.Button(controls_frame, text="Cargar Datos", command=self.fetch_data)
        self.fetch_btn.grid(row=0, column=4, padx=(15, 5))

        self.export_btn = ttk.Button(controls_frame, text="Exportar CSV", command=self.export_csv)
        self.export_btn.grid(row=0, column=5)
        self.export_btn.state(["disabled"])

        # Tabs principales
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Tab 1: Gráficos de Precios
        self._create_price_charts_tab()

        # Tab 2: Tabla de Datos
        self._create_data_table_tab()

        # Tab 3: Ratios Financieros
        self._create_financial_ratios_tab()

        # Tab 4: Predicción ML
        self._create_prediction_tab()

        # Tab 5: Ranking de Portfolio
        self._create_portfolio_ranking_tab()

        # Barra de estado
        self.status_var = tk.StringVar(value="Listo")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_price_charts_tab(self):
        """Crea el tab de gráficos de precios"""
        self.tab_charts = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_charts, text="Gráficos de Precios")

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.ax_price = self.figure.add_subplot(211)
        self.ax_volume = self.figure.add_subplot(212, sharex=self.ax_price)
        self.figure.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.tab_charts)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    def _create_data_table_tab(self):
        """Crea el tab de tabla de datos"""
        self.tab_table = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_table, text="Tabla de Datos")

        # Treeview con scrollbar
        tree_frame = ttk.Frame(self.tab_table)
        tree_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("Date", "Open", "High", "Low", "Close", "Volume"),
            show="headings",
            yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.tree.yview)

        for col in ("Date", "Open", "High", "Low", "Close", "Volume"):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150, anchor=tk.CENTER)

        self.tree.pack(expand=True, fill=tk.BOTH)

    def _create_financial_ratios_tab(self):
        """Crea el tab de ratios financieros"""
        self.tab_ratios = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_ratios, text="Ratios Financieros")

        # Frame con scrollbar
        canvas = tk.Canvas(self.tab_ratios)
        scrollbar = ttk.Scrollbar(self.tab_ratios, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Botón para cargar ratios
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(pady=10)

        self.load_ratios_btn = ttk.Button(
            btn_frame,
            text="Cargar Ratios Financieros",
            command=self.load_financial_ratios
        )
        self.load_ratios_btn.pack()

        # Text widget para mostrar ratios
        self.ratios_text = scrolledtext.ScrolledText(
            scrollable_frame,
            width=100,
            height=35,
            font=("Consolas", 10)
        )
        self.ratios_text.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _create_prediction_tab(self):
        """Crea el tab de predicción ML"""
        self.tab_prediction = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_prediction, text="Predicción ML")

        # Controles
        controls = ttk.Frame(self.tab_prediction, padding=10)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Modelo:").grid(row=0, column=0, padx=5)
        self.model_var = tk.StringVar(value="KNN")
        model_combo = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=["KNN", "Random Forest", "LSTM", "Neural Network"],
            state="readonly",
            width=20
        )
        model_combo.grid(row=0, column=1, padx=5)

        ttk.Label(controls, text="Look Back (días):").grid(row=0, column=2, padx=(20, 5))
        self.lookback_var = tk.StringVar(value="30")
        lookback_spin = ttk.Spinbox(
            controls,
            textvariable=self.lookback_var,
            from_=10,
            to=200,
            width=10,
            increment=10
        )
        lookback_spin.grid(row=0, column=3, padx=5)

        # Tooltip label
        ttk.Label(controls, text="(Recomendado: 30-50)", font=("", 8)).grid(row=0, column=4, sticky=tk.W, padx=5)

        self.predict_btn = ttk.Button(controls, text="Entrenar y Predecir", command=self.run_prediction)
        self.predict_btn.grid(row=0, column=5, padx=20)
        self.predict_btn.state(["disabled"])

        # Área de resultados
        results_frame = ttk.Frame(self.tab_prediction)
        results_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.prediction_text = scrolledtext.ScrolledText(
            results_frame,
            width=100,
            height=35,
            font=("Consolas", 10)
        )
        self.prediction_text.pack(expand=True, fill=tk.BOTH)

    def _create_portfolio_ranking_tab(self):
        """Crea el tab de ranking de portfolio"""
        self.tab_ranking = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_ranking, text="Ranking Portfolio")

        # Controles
        controls = ttk.Frame(self.tab_ranking, padding=10)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Tickers (separados por coma):").grid(row=0, column=0, padx=5)
        self.portfolio_tickers_var = tk.StringVar(value="AAPL,MSFT,GOOGL,AMZN,TSLA,META")
        portfolio_entry = ttk.Entry(controls, textvariable=self.portfolio_tickers_var, width=50)
        portfolio_entry.grid(row=0, column=1, padx=5)

        rank_btn = ttk.Button(controls, text="Calcular Ranking", command=self.calculate_portfolio_ranking)
        rank_btn.grid(row=0, column=2, padx=20)

        # Treeview para ranking
        tree_frame = ttk.Frame(self.tab_ranking)
        tree_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.ranking_tree = ttk.Treeview(
            tree_frame,
            columns=("Rank", "Symbol", "Name", "PE", "ROA", "Price", "MarketCap", "MagicRank"),
            show="headings",
            yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.ranking_tree.yview)

        headers = {
            "Rank": "Rank",
            "Symbol": "Symbol",
            "Name": "Nombre",
            "PE": "P/E Ratio",
            "ROA": "ROA (%)",
            "Price": "Precio ($)",
            "MarketCap": "Cap. Mercado ($B)",
            "MagicRank": "Magic Rank"
        }

        for col, header in headers.items():
            self.ranking_tree.heading(col, text=header)
            width = 120 if col != "Name" else 200
            self.ranking_tree.column(col, width=width, anchor=tk.CENTER)

        self.ranking_tree.pack(expand=True, fill=tk.BOTH)

    def fetch_data(self):
        """Descarga datos de la acción"""
        ticker = self.ticker_var.get().strip().upper()
        period_label = self.period_var.get()
        period = PERIOD_OPTIONS.get(period_label, "1y")

        if not ticker:
            messagebox.showwarning("Validación", "Ingresa un ticker válido (ej: MSFT, AAPL)")
            return

        self.status_var.set(f"Descargando {ticker} ({period_label})...")
        self.update_idletasks()

        def download_thread():
            try:
                # Descargar datos históricos
                data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
                if data.empty:
                    raise ValueError(f"No se encontraron datos para {ticker}")

                # Aplanar MultiIndex si existe (yfinance 0.2.36+)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                data = data.reset_index()
                data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    data[col] = pd.to_numeric(data[col], errors="coerce")

                data = data.dropna(subset=["Date"]).copy()
                # Forward fill para datos faltantes, luego rellenar con 0
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    data[col] = data[col].ffill().fillna(0)

                self.df_prices = data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

                # Obtener información del ticker
                ticker_obj = yf.Ticker(ticker)
                self.ticker_info = ticker_obj.info

                # Actualizar UI en el hilo principal
                self.after(0, self._update_ui_after_fetch)

            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda: self._show_error(error_msg))

        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

    def _update_ui_after_fetch(self):
        """Actualiza la UI después de descargar datos"""
        self._update_price_charts()
        self._update_data_table()
        self.export_btn.state(["!disabled"])
        self.predict_btn.state(["!disabled"])
        self.status_var.set(f"{len(self.df_prices)} registros cargados para {self.ticker_var.get()}")

    def _show_error(self, error_msg):
        """Muestra un error"""
        self.status_var.set("Error")
        messagebox.showerror("Error", error_msg)

    def _update_price_charts(self):
        """Actualiza los gráficos de precios"""
        if self.df_prices.empty:
            return

        self.ax_price.clear()
        self.ax_volume.clear()

        import matplotlib.dates as mdates

        try:
            dates_ts = pd.to_datetime(self.df_prices["Date"], errors="coerce").dropna()
            try:
                dates_ts = dates_ts.dt.tz_localize(None)
            except:
                pass

            dates_py = [d.to_pydatetime() for d in dates_ts]
            dates_x = mdates.date2num(dates_py)

            # Convertir y rellenar datos faltantes
            close_s = pd.to_numeric(
                self.df_prices.loc[dates_ts.index, "Close"], errors="coerce"
            )
            close_s = close_s.ffill().fillna(0.0).astype(float)

            vol_s = pd.to_numeric(
                self.df_prices.loc[dates_ts.index, "Volume"], errors="coerce"
            )
            vol_s = vol_s.ffill().fillna(0.0).astype(float)

            min_len = min(len(dates_x), len(close_s), len(vol_s))
            if min_len == 0:
                return

            dates_x = dates_x[-min_len:]
            close_y = close_s.to_numpy()[-min_len:]
            vol_y = vol_s.to_numpy()[-min_len:]

            # Gráfico de precio
            self.ax_price.plot_date(dates_x, close_y, fmt='-', label="Close", color="tab:blue")
            self.ax_price.set_title(f"{self.ticker_var.get()} - Precio de Cierre")
            self.ax_price.set_ylabel("USD")
            self.ax_price.grid(True, alpha=0.3)
            self.ax_price.legend(loc="upper left")

            # Gráfico de volumen
            self.ax_volume.bar(dates_x, vol_y, color="tab:gray")
            self.ax_volume.set_title("Volumen")
            self.ax_volume.set_ylabel("Shares")
            self.ax_volume.grid(True, alpha=0.3)

            self.ax_volume.xaxis_date()
            self.figure.autofmt_xdate()
            self.canvas.draw_idle()

        except Exception as e:
            # Fallback con índices
            x = list(range(len(self.df_prices)))
            y_close = pd.to_numeric(self.df_prices["Close"], errors="coerce").fillna(0.0).astype(float).tolist()
            y_vol = pd.to_numeric(self.df_prices["Volume"], errors="coerce").fillna(0.0).astype(float).tolist()

            min_len = min(len(x), len(y_close), len(y_vol))
            if min_len == 0:
                return

            self.ax_price.plot(x[-min_len:], y_close[-min_len:], label="Close", color="tab:blue")
            self.ax_price.set_title(f"{self.ticker_var.get()} - Precio de Cierre")
            self.ax_price.set_ylabel("USD")
            self.ax_price.grid(True, alpha=0.3)

            self.ax_volume.bar(x[-min_len:], y_vol[-min_len:], color="tab:gray")
            self.ax_volume.set_title("Volumen")
            self.ax_volume.set_ylabel("Shares")
            self.ax_volume.grid(True, alpha=0.3)

            self.canvas.draw_idle()

    def _update_data_table(self):
        """Actualiza la tabla de datos"""
        for row in self.tree.get_children():
            self.tree.delete(row)

        def fmt_float(x):
            """Formatea un número como float con 2 decimales"""
            try:
                if pd.isna(x):
                    return "0.00"
                return f"{float(x):.2f}"
            except (ValueError, TypeError):
                return "0.00"

        def fmt_int(x):
            """Formatea un número como entero con separador de miles"""
            try:
                if pd.isna(x):
                    return "0"
                v = int(float(x))
                return f"{v:,}"
            except (ValueError, TypeError):
                return "0"

        def fmt_date(x):
            """Formatea una fecha como string YYYY-MM-DD"""
            try:
                if pd.isna(x):
                    return ""
                # Si ya es un Timestamp de pandas, usar directamente
                if isinstance(x, pd.Timestamp):
                    return x.strftime("%Y-%m-%d")
                # Si no, intentar convertir
                date_obj = pd.to_datetime(x, errors='coerce')
                if pd.isna(date_obj):
                    return ""
                return date_obj.strftime("%Y-%m-%d")
            except (ValueError, TypeError, AttributeError):
                return ""

        # Obtener últimos 200 registros de forma segura
        df_display = self.df_prices.tail(200) if len(self.df_prices) > 0 else self.df_prices

        for _, row in df_display.iterrows():
            try:
                self.tree.insert("", tk.END, values=(
                    fmt_date(row.get("Date", "")),
                    fmt_float(row.get('Open', 0)),
                    fmt_float(row.get('High', 0)),
                    fmt_float(row.get('Low', 0)),
                    fmt_float(row.get('Close', 0)),
                    fmt_int(row.get('Volume', 0))
                ))
            except Exception as e:
                print(f"Error insertando fila: {e}")
                continue

    def load_financial_ratios(self):
        """Carga los ratios financieros"""
        ticker = self.ticker_var.get().strip().upper()

        if not ticker:
            messagebox.showwarning("Validación", "Primero carga datos de un ticker")
            return

        self.status_var.set(f"Cargando ratios financieros de {ticker}...")
        self.ratios_text.delete(1.0, tk.END)
        self.ratios_text.insert(tk.END, "Cargando...\n")
        self.update_idletasks()

        def load_thread():
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info

                ratios = {
                    'Symbol': ticker,
                    'Company Name': info.get('shortName', 'N/A'),
                    'P/E Ratio': info.get('trailingPE', None),
                    'Forward P/E': info.get('forwardPE', None),
                    'P/B Ratio': info.get('priceToBook', None),
                    'P/S Ratio': info.get('priceToSalesTrailing12Months', None),
                    'PEG Ratio': info.get('pegRatio', None),
                    'Profit Margin': info.get('profitMargins', None),
                    'Operating Margin': info.get('operatingMargins', None),
                    'ROA (%)': info.get('returnOnAssets', None),
                    'ROE (%)': info.get('returnOnEquity', None),
                    'Current Ratio': info.get('currentRatio', None),
                    'Quick Ratio': info.get('quickRatio', None),
                    'Debt to Equity': info.get('debtToEquity', None),
                    'Total Debt': info.get('totalDebt', None),
                    'Dividend Yield': info.get('dividendYield', None),
                    'Payout Ratio': info.get('payoutRatio', None),
                    'Market Cap': info.get('marketCap', None),
                    'Enterprise Value': info.get('enterpriseValue', None),
                    'Beta': info.get('beta', None),
                    'Current Price': info.get('currentPrice', None),
                    '52 Week High': info.get('fiftyTwoWeekHigh', None),
                    '52 Week Low': info.get('fiftyTwoWeekLow', None),
                }

                self.financial_ratios = ratios
                self.after(0, self._display_financial_ratios)

            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda: self._show_error(error_msg))

        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()

    def _display_financial_ratios(self):
        """Muestra los ratios financieros"""
        self.ratios_text.delete(1.0, tk.END)

        ratios = self.financial_ratios

        output = "=" * 80 + "\n"
        output += f"RATIOS FINANCIEROS - {ratios['Symbol']}\n"
        output += f"Empresa: {ratios['Company Name']}\n"
        output += "=" * 80 + "\n\n"

        # Información General
        output += "INFORMACIÓN GENERAL:\n"
        output += f"  Precio Actual: ${ratios['Current Price']:.2f}\n" if ratios['Current Price'] else "  Precio Actual: N/A\n"
        output += f"  Market Cap: ${ratios['Market Cap']/1e9:.2f}B\n" if ratios['Market Cap'] else "  Market Cap: N/A\n"
        output += f"  Beta: {ratios['Beta']:.2f}\n" if ratios['Beta'] else "  Beta: N/A\n"
        output += f"  52W High: ${ratios['52 Week High']:.2f}\n" if ratios['52 Week High'] else "  52W High: N/A\n"
        output += f"  52W Low: ${ratios['52 Week Low']:.2f}\n\n" if ratios['52 Week Low'] else "  52W Low: N/A\n\n"

        # Ratios de Valoración
        output += "RATIOS DE VALORACIÓN:\n"
        output += f"  P/E Ratio: {ratios['P/E Ratio']:.2f}\n" if ratios['P/E Ratio'] else "  P/E Ratio: N/A\n"
        output += f"  Forward P/E: {ratios['Forward P/E']:.2f}\n" if ratios['Forward P/E'] else "  Forward P/E: N/A\n"
        output += f"  P/B Ratio: {ratios['P/B Ratio']:.2f}\n" if ratios['P/B Ratio'] else "  P/B Ratio: N/A\n"
        output += f"  P/S Ratio: {ratios['P/S Ratio']:.2f}\n" if ratios['P/S Ratio'] else "  P/S Ratio: N/A\n"
        output += f"  PEG Ratio: {ratios['PEG Ratio']:.2f}\n\n" if ratios['PEG Ratio'] else "  PEG Ratio: N/A\n\n"

        # Ratios de Rentabilidad
        output += "RATIOS DE RENTABILIDAD:\n"
        output += f"  Profit Margin: {ratios['Profit Margin']*100:.2f}%\n" if ratios['Profit Margin'] else "  Profit Margin: N/A\n"
        output += f"  Operating Margin: {ratios['Operating Margin']*100:.2f}%\n" if ratios['Operating Margin'] else "  Operating Margin: N/A\n"
        output += f"  ROA: {ratios['ROA (%)']*100:.2f}%\n" if ratios['ROA (%)'] else "  ROA: N/A\n"
        output += f"  ROE: {ratios['ROE (%)']*100:.2f}%\n\n" if ratios['ROE (%)'] else "  ROE: N/A\n\n"

        # Ratios de Liquidez
        output += "RATIOS DE LIQUIDEZ:\n"
        output += f"  Current Ratio: {ratios['Current Ratio']:.2f}\n" if ratios['Current Ratio'] else "  Current Ratio: N/A\n"
        output += f"  Quick Ratio: {ratios['Quick Ratio']:.2f}\n\n" if ratios['Quick Ratio'] else "  Quick Ratio: N/A\n\n"

        # Ratios de Deuda
        output += "RATIOS DE DEUDA:\n"
        output += f"  Debt to Equity: {ratios['Debt to Equity']:.2f}\n" if ratios['Debt to Equity'] else "  Debt to Equity: N/A\n"
        output += f"  Total Debt: ${ratios['Total Debt']/1e9:.2f}B\n\n" if ratios['Total Debt'] else "  Total Debt: N/A\n\n"

        # Dividendos
        output += "DIVIDENDOS:\n"
        output += f"  Dividend Yield: {ratios['Dividend Yield']*100:.2f}%\n" if ratios['Dividend Yield'] else "  Dividend Yield: N/A\n"
        output += f"  Payout Ratio: {ratios['Payout Ratio']*100:.2f}%\n" if ratios['Payout Ratio'] else "  Payout Ratio: N/A\n"

        output += "\n" + "=" * 80 + "\n"

        # Análisis de Salud Financiera
        output += "\nANÁLISIS DE SALUD FINANCIERA:\n"
        output += "=" * 80 + "\n"

        score = 0
        max_score = 0

        # Valoración
        if ratios['P/E Ratio']:
            max_score += 1
            if ratios['P/E Ratio'] < 15:
                output += "Valoración: Subvalorada (P/E < 15)\n"
                score += 1
            elif ratios['P/E Ratio'] < 25:
                output += "Valoración: Justa (P/E 15-25)\n"
                score += 0.5
            else:
                output += "Valoración: Sobrevalorada (P/E > 25)\n"

        # Rentabilidad
        if ratios['ROE (%)']:
            max_score += 1
            roe_pct = ratios['ROE (%)'] * 100
            if roe_pct > 15:
                output += f"Rentabilidad: Excelente (ROE {roe_pct:.1f}%)\n"
                score += 1
            elif roe_pct > 10:
                output += f"Rentabilidad: Buena (ROE {roe_pct:.1f}%)\n"
                score += 0.5
            else:
                output += f"Rentabilidad: Baja (ROE {roe_pct:.1f}%)\n"

        # Liquidez
        if ratios['Current Ratio']:
            max_score += 1
            if ratios['Current Ratio'] > 1.5:
                output += f"Liquidez: Buena (Current Ratio {ratios['Current Ratio']:.2f})\n"
                score += 1
            elif ratios['Current Ratio'] > 1.0:
                output += f"Liquidez: Adecuada (Current Ratio {ratios['Current Ratio']:.2f})\n"
                score += 0.5
            else:
                output += f"Liquidez: Baja (Current Ratio {ratios['Current Ratio']:.2f})\n"

        # Deuda
        if ratios['Debt to Equity']:
            max_score += 1
            debt_equity = ratios['Debt to Equity'] / 100
            if debt_equity < 0.5:
                output += f"Deuda: Baja (D/E {debt_equity:.2f})\n"
                score += 1
            elif debt_equity < 1.0:
                output += f"Deuda: Moderada (D/E {debt_equity:.2f})\n"
                score += 0.5
            else:
                output += f"Deuda: Alta (D/E {debt_equity:.2f})\n"

        if max_score > 0:
            overall_score = (score / max_score) * 100
            output += f"\nPuntuación General: {overall_score:.1f}/100\n"

        output += "=" * 80 + "\n"

        self.ratios_text.insert(tk.END, output)
        self.status_var.set(f"Ratios financieros cargados para {self.ticker_var.get()}")

    def run_prediction(self):
        """Ejecuta la predicción ML"""
        if self.df_prices.empty:
            messagebox.showwarning("Validación", "Primero carga datos de un ticker")
            return

        model_name = self.model_var.get()
        try:
            look_back = int(self.lookback_var.get())
        except:
            messagebox.showerror("Error", "Look Back debe ser un número entero")
            return

        # Validar datos suficientes
        total_records = len(self.df_prices)
        train_size = int(total_records * 0.8)
        test_size = total_records - train_size

        min_test_needed = look_back + 10  # Mínimo 10 registros para predicción

        if test_size < min_test_needed:
            max_lookback = max(30, test_size - 10)
            messagebox.showerror(
                "Datos Insuficientes",
                f"No hay suficientes datos para Look Back = {look_back}\n\n"
                f"Datos totales: {total_records}\n"
                f"Datos de prueba: {test_size}\n"
                f"Necesario: {min_test_needed}\n\n"
                f"SOLUCIONES:\n"
                f"1. Reducir Look Back a {max_lookback} o menos\n"
                f"2. Cargar un período más largo (2 años o 5 años)\n"
                f"3. Usar período 'Todo' para datos históricos completos"
            )
            return

        self.status_var.set(f"Entrenando modelo {model_name}...")
        self.prediction_text.delete(1.0, tk.END)
        self.prediction_text.insert(tk.END, f"Entrenando modelo {model_name}...\n")
        self.update_idletasks()

        def prediction_thread():
            try:
                # Preparar datos
                prices = self.df_prices[['Close']].values.astype(float)

                # Dividir en train/test
                train_size = int(len(prices) * 0.8)
                train_data = prices[:train_size]
                test_data = prices[train_size:]

                # Escalar
                scaler = MinMaxScaler(feature_range=(0, 1))
                train_scaled = scaler.fit_transform(train_data)
                test_scaled = scaler.transform(test_data)

                # Crear datasets
                def create_dataset(data, look_back):
                    X, Y = [], []
                    for i in range(len(data) - look_back):
                        X.append(data[i:(i + look_back), 0])
                        Y.append(data[i + look_back, 0])
                    return np.array(X), np.array(Y)

                X_train, y_train = create_dataset(train_scaled, look_back)
                X_test, y_test = create_dataset(test_scaled, look_back)

                # Crear directorio para modelos
                os.makedirs('models_saved', exist_ok=True)

                # Entrenar modelo según selección
                if model_name == "KNN":
                    model = KNNModel(n_neighbors=5)
                    model.train(X_train, y_train)
                    predictions = model.predict(X_test)

                elif model_name == "Random Forest":
                    model = RandomForestModel(n_estimators=100, max_depth=10)
                    model.train(X_train, y_train)
                    predictions = model.predict(X_test)

                elif model_name == "LSTM":
                    model = LSTMModel(look_back=look_back, lstm_units=32, dropout_rate=0.2)
                    model.train(X_train, y_train, epochs=20, batch_size=64)
                    predictions = model.predict(X_test)

                elif model_name == "Neural Network":
                    model = NeuralNetworkModel(input_dim=look_back)
                    model.train(X_train, y_train, epochs=20, batch_size=64)
                    predictions = model.predict(X_test)

                # Desescalar
                predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
                y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

                # Calcular métricas
                mae, mse, rmse = model.evaluate(y_test_inverse, predictions)

                # Guardar resultados
                self.predictions = {
                    'model_name': model_name,
                    'predictions': predictions,
                    'actual': y_test_inverse,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'look_back': look_back
                }

                self.after(0, self._display_prediction_results)

            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda: self._show_prediction_error(error_msg))

        thread = threading.Thread(target=prediction_thread, daemon=True)
        thread.start()

    def _display_prediction_results(self):
        """Muestra los resultados de la predicción"""
        self.prediction_text.delete(1.0, tk.END)

        pred = self.predictions

        output = "=" * 80 + "\n"
        output += f"RESULTADOS DE PREDICCIÓN - {pred['model_name']}\n"
        output += "=" * 80 + "\n\n"
        output += f"Modelo: {pred['model_name']}\n"
        output += f"Look Back: {pred['look_back']} días\n"
        output += f"Datos de prueba: {len(pred['actual'])} registros\n\n"

        output += "MÉTRICAS DE EVALUACIÓN:\n"
        output += "-" * 80 + "\n"
        output += f"Mean Absolute Error (MAE):  ${pred['mae']:.2f}\n"
        output += f"Mean Squared Error (MSE):   ${pred['mse']:.2f}\n"
        output += f"Root Mean Squared Error:    ${pred['rmse']:.2f}\n"
        output += "-" * 80 + "\n\n"

        output += "ÚLTIMAS 10 PREDICCIONES:\n"
        output += "-" * 80 + "\n"
        output += f"{'Índice':<10} {'Real ($)':<15} {'Predicho ($)':<15} {'Error ($)':<15}\n"
        output += "-" * 80 + "\n"

        for i in range(min(10, len(pred['actual']))):
            idx = len(pred['actual']) - 10 + i
            real = pred['actual'][idx][0]
            pred_val = pred['predictions'][idx][0]
            error = abs(real - pred_val)
            output += f"{idx:<10} ${real:<14.2f} ${pred_val:<14.2f} ${error:<14.2f}\n"

        output += "=" * 80 + "\n"

        self.prediction_text.insert(tk.END, output)
        self.status_var.set(f"Predicción completada - MAE: ${pred['mae']:.2f}")

    def _show_prediction_error(self, error_msg):
        """Muestra un error de predicción"""
        self.prediction_text.delete(1.0, tk.END)
        self.prediction_text.insert(tk.END, f"ERROR: {error_msg}\n")
        self.status_var.set("Error en predicción")

    def calculate_portfolio_ranking(self):
        """Calcula el ranking del portfolio"""
        tickers_str = self.portfolio_tickers_var.get().strip()

        if not tickers_str:
            messagebox.showwarning("Validación", "Ingresa al menos un ticker")
            return

        tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]

        self.status_var.set(f"Calculando ranking para {len(tickers)} acciones...")
        self.update_idletasks()

        # Limpiar tabla
        for row in self.ranking_tree.get_children():
            self.ranking_tree.delete(row)

        def ranking_thread():
            try:
                data_list = []

                for ticker in tickers:
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        info = ticker_obj.info

                        data_list.append({
                            'symbol': ticker,
                            'shortName': info.get('shortName', 'N/A'),
                            'trailingPE': info.get('trailingPE', None),
                            'returnOnAssets': info.get('returnOnAssets', None),
                            'currentPrice': info.get('currentPrice', None),
                            'marketCap': info.get('marketCap', None)
                        })
                    except:
                        continue

                df = pd.DataFrame(data_list)

                if df.empty:
                    raise ValueError("No se pudieron obtener datos")

                # Filtrar datos válidos
                df_clean = df.dropna(subset=['trailingPE', 'returnOnAssets']).copy()

                if df_clean.empty:
                    raise ValueError("No hay datos suficientes para calcular ranking")

                # Calcular rankings
                df_clean['PE Rank'] = df_clean['trailingPE'].rank(ascending=True)
                df_clean['ROA Rank'] = df_clean['returnOnAssets'].rank(ascending=False)
                df_clean['Magic Rank'] = df_clean['PE Rank'] + df_clean['ROA Rank']

                # Ordenar
                df_sorted = df_clean.sort_values(by='Magic Rank', ascending=True).reset_index(drop=True)

                self.after(0, lambda: self._display_portfolio_ranking(df_sorted))

            except Exception as e:
                error_msg = f"Error en ranking: {str(e)}"
                self.after(0, lambda: self._show_error(error_msg))

        thread = threading.Thread(target=ranking_thread, daemon=True)
        thread.start()

    def _display_portfolio_ranking(self, df):
        """Muestra el ranking del portfolio"""
        for row in self.ranking_tree.get_children():
            self.ranking_tree.delete(row)

        for idx, row in df.iterrows():
            rank = idx + 1
            symbol = row['symbol']
            name = row.get('shortName', 'N/A')[:30]
            pe = f"{row['trailingPE']:.2f}" if pd.notna(row['trailingPE']) else "N/A"
            roa = f"{row['returnOnAssets']*100:.2f}" if pd.notna(row['returnOnAssets']) else "N/A"
            price = f"{row['currentPrice']:.2f}" if pd.notna(row['currentPrice']) else "N/A"
            mcap = f"{row['marketCap']/1e9:.2f}" if pd.notna(row['marketCap']) else "N/A"
            magic_rank = f"{row['Magic Rank']:.1f}" if pd.notna(row['Magic Rank']) else "N/A"

            self.ranking_tree.insert("", tk.END, values=(
                rank, symbol, name, pe, roa, price, mcap, magic_rank
            ))

        self.status_var.set(f"Ranking calculado para {len(df)} acciones")

    def export_csv(self):
        """Exporta los datos a CSV"""
        if self.df_prices.empty:
            messagebox.showinfo("Exportar", "No hay datos para exportar")
            return

        file = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"{self.ticker_var.get()}_{PERIOD_OPTIONS[self.period_var.get()]}.csv"
        )

        if file:
            self.df_prices.to_csv(file, index=False)
            messagebox.showinfo("Exportar", f"Datos guardados en:\n{file}")


def main():
    """Función principal"""
    app = FinancialAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
