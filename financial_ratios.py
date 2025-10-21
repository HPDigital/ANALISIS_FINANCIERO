"""
Análisis de Ratios Financieros
Obtiene y analiza métricas financieras clave de empresas
"""

import yfinance as yf
import pandas as pd


def get_financial_ratios(ticker_symbol):
    """
    Obtiene ratios financieros clave de una empresa
    
    Args:
        ticker_symbol (str): Símbolo de la acción
        
    Returns:
        dict: Diccionario con ratios financieros
    """
    print(f"\n🔍 Obteniendo ratios financieros de {ticker_symbol}...")
    
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    
    ratios = {
        'Symbol': ticker_symbol,
        'Company Name': info.get('shortName', 'N/A'),
        
        # Ratios de Valoración
        'P/E Ratio': info.get('trailingPE', None),
        'Forward P/E': info.get('forwardPE', None),
        'P/B Ratio': info.get('priceToBook', None),
        'P/S Ratio': info.get('priceToSalesTrailing12Months', None),
        'PEG Ratio': info.get('pegRatio', None),
        
        # Ratios de Rentabilidad
        'Profit Margin': info.get('profitMargins', None),
        'Operating Margin': info.get('operatingMargins', None),
        'ROA (%)': info.get('returnOnAssets', None),
        'ROE (%)': info.get('returnOnEquity', None),
        
        # Ratios de Liquidez
        'Current Ratio': info.get('currentRatio', None),
        'Quick Ratio': info.get('quickRatio', None),
        
        # Ratios de Deuda
        'Debt to Equity': info.get('debtToEquity', None),
        'Total Debt': info.get('totalDebt', None),
        
        # Dividendos
        'Dividend Yield': info.get('dividendYield', None),
        'Payout Ratio': info.get('payoutRatio', None),
        
        # Información General
        'Market Cap': info.get('marketCap', None),
        'Enterprise Value': info.get('enterpriseValue', None),
        'Beta': info.get('beta', None),
        'Current Price': info.get('currentPrice', None),
        '52 Week High': info.get('fiftyTwoWeekHigh', None),
        '52 Week Low': info.get('fiftyTwoWeekLow', None),
    }
    
    return ratios


def analyze_financial_health(ratios):
    """
    Analiza la salud financiera basándose en los ratios
    
    Args:
        ratios (dict): Diccionario con ratios financieros
        
    Returns:
        dict: Análisis de salud financiera
    """
    analysis = {
        'Symbol': ratios['Symbol'],
        'Overall Score': 0,
        'Valoración': 'N/A',
        'Rentabilidad': 'N/A',
        'Liquidez': 'N/A',
        'Deuda': 'N/A'
    }
    
    score = 0
    max_score = 0
    
    # Análisis de Valoración (P/E Ratio)
    if ratios['P/E Ratio']:
        max_score += 1
        if ratios['P/E Ratio'] < 15:
            analysis['Valoración'] = '✅ Subvalorada'
            score += 1
        elif ratios['P/E Ratio'] < 25:
            analysis['Valoración'] = '⚠️ Valoración Justa'
            score += 0.5
        else:
            analysis['Valoración'] = '❌ Sobrevalorada'
    
    # Análisis de Rentabilidad (ROE)
    if ratios['ROE (%)']:
        max_score += 1
        roe_pct = ratios['ROE (%)'] * 100
        if roe_pct > 15:
            analysis['Rentabilidad'] = f'✅ Excelente ({roe_pct:.1f}%)'
            score += 1
        elif roe_pct > 10:
            analysis['Rentabilidad'] = f'⚠️ Buena ({roe_pct:.1f}%)'
            score += 0.5
        else:
            analysis['Rentabilidad'] = f'❌ Baja ({roe_pct:.1f}%)'
    
    # Análisis de Liquidez (Current Ratio)
    if ratios['Current Ratio']:
        max_score += 1
        if ratios['Current Ratio'] > 1.5:
            analysis['Liquidez'] = f'✅ Buena ({ratios["Current Ratio"]:.2f})'
            score += 1
        elif ratios['Current Ratio'] > 1.0:
            analysis['Liquidez'] = f'⚠️ Adecuada ({ratios["Current Ratio"]:.2f})'
            score += 0.5
        else:
            analysis['Liquidez'] = f'❌ Baja ({ratios["Current Ratio"]:.2f})'
    
    # Análisis de Deuda (Debt to Equity)
    if ratios['Debt to Equity']:
        max_score += 1
        debt_equity = ratios['Debt to Equity'] / 100  # Convertir de porcentaje
        if debt_equity < 0.5:
            analysis['Deuda'] = f'✅ Baja ({debt_equity:.2f})'
            score += 1
        elif debt_equity < 1.0:
            analysis['Deuda'] = f'⚠️ Moderada ({debt_equity:.2f})'
            score += 0.5
        else:
            analysis['Deuda'] = f'❌ Alta ({debt_equity:.2f})'
    
    # Calcular score final
    if max_score > 0:
        analysis['Overall Score'] = (score / max_score) * 100
    
    return analysis


def display_financial_ratios(ratios):
    """
    Muestra los ratios financieros de forma organizada
    
    Args:
        ratios (dict): Diccionario con ratios financieros
    """
    print("\n" + "="*80)
    print(f"📊 RATIOS FINANCIEROS - {ratios['Symbol']}")
    print(f"Empresa: {ratios['Company Name']}")
    print("="*80 + "\n")
    
    # Información General
    print("💼 INFORMACIÓN GENERAL:")
    print(f"  Precio Actual: ${ratios['Current Price']:.2f}" if ratios['Current Price'] else "  Precio Actual: N/A")
    print(f"  Market Cap: ${ratios['Market Cap']/1e9:.2f}B" if ratios['Market Cap'] else "  Market Cap: N/A")
    print(f"  Beta: {ratios['Beta']:.2f}" if ratios['Beta'] else "  Beta: N/A")
    print(f"  52W High: ${ratios['52 Week High']:.2f}" if ratios['52 Week High'] else "  52W High: N/A")
    print(f"  52W Low: ${ratios['52 Week Low']:.2f}" if ratios['52 Week Low'] else "  52W Low: N/A")
    
    # Ratios de Valoración
    print("\n💰 RATIOS DE VALORACIÓN:")
    print(f"  P/E Ratio: {ratios['P/E Ratio']:.2f}" if ratios['P/E Ratio'] else "  P/E Ratio: N/A")
    print(f"  Forward P/E: {ratios['Forward P/E']:.2f}" if ratios['Forward P/E'] else "  Forward P/E: N/A")
    print(f"  P/B Ratio: {ratios['P/B Ratio']:.2f}" if ratios['P/B Ratio'] else "  P/B Ratio: N/A")
    print(f"  P/S Ratio: {ratios['P/S Ratio']:.2f}" if ratios['P/S Ratio'] else "  P/S Ratio: N/A")
    print(f"  PEG Ratio: {ratios['PEG Ratio']:.2f}" if ratios['PEG Ratio'] else "  PEG Ratio: N/A")
    
    # Ratios de Rentabilidad
    print("\n📈 RATIOS DE RENTABILIDAD:")
    print(f"  Profit Margin: {ratios['Profit Margin']*100:.2f}%" if ratios['Profit Margin'] else "  Profit Margin: N/A")
    print(f"  Operating Margin: {ratios['Operating Margin']*100:.2f}%" if ratios['Operating Margin'] else "  Operating Margin: N/A")
    print(f"  ROA: {ratios['ROA (%)']*100:.2f}%" if ratios['ROA (%)'] else "  ROA: N/A")
    print(f"  ROE: {ratios['ROE (%)']*100:.2f}%" if ratios['ROE (%)'] else "  ROE: N/A")
    
    # Ratios de Liquidez
    print("\n💧 RATIOS DE LIQUIDEZ:")
    print(f"  Current Ratio: {ratios['Current Ratio']:.2f}" if ratios['Current Ratio'] else "  Current Ratio: N/A")
    print(f"  Quick Ratio: {ratios['Quick Ratio']:.2f}" if ratios['Quick Ratio'] else "  Quick Ratio: N/A")
    
    # Ratios de Deuda
    print("\n💳 RATIOS DE DEUDA:")
    print(f"  Debt to Equity: {ratios['Debt to Equity']:.2f}" if ratios['Debt to Equity'] else "  Debt to Equity: N/A")
    print(f"  Total Debt: ${ratios['Total Debt']/1e9:.2f}B" if ratios['Total Debt'] else "  Total Debt: N/A")
    
    # Dividendos
    print("\n💵 DIVIDENDOS:")
    print(f"  Dividend Yield: {ratios['Dividend Yield']*100:.2f}%" if ratios['Dividend Yield'] else "  Dividend Yield: N/A")
    print(f"  Payout Ratio: {ratios['Payout Ratio']*100:.2f}%" if ratios['Payout Ratio'] else "  Payout Ratio: N/A")
    
    print("\n" + "="*80 + "\n")


def main():
    """Función principal para análisis de ratios financieros"""
    
    # Lista de empresas para analizar
    companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    all_ratios = []
    all_analysis = []
    
    for company in companies:
        # Obtener ratios
        ratios = get_financial_ratios(company)
        all_ratios.append(ratios)
        
        # Mostrar ratios
        display_financial_ratios(ratios)
        
        # Analizar salud financiera
        analysis = analyze_financial_health(ratios)
        all_analysis.append(analysis)
    
    # Crear DataFrame con todos los resultados
    df_ratios = pd.DataFrame(all_ratios)
    df_analysis = pd.DataFrame(all_analysis)
    
    # Guardar resultados
    df_ratios.to_csv('financial_ratios_results.csv', index=False)
    df_analysis.to_csv('financial_health_analysis.csv', index=False)
    
    # Mostrar resumen de análisis
    print("\n" + "="*80)
    print("🏥 RESUMEN DE SALUD FINANCIERA")
    print("="*80 + "\n")
    
    df_analysis_sorted = df_analysis.sort_values('Overall Score', ascending=False)
    
    for idx, row in df_analysis_sorted.iterrows():
        print(f"{row['Symbol']} - Score: {row['Overall Score']:.1f}%")
        print(f"  Valoración: {row['Valoración']}")
        print(f"  Rentabilidad: {row['Rentabilidad']}")
        print(f"  Liquidez: {row['Liquidez']}")
        print(f"  Deuda: {row['Deuda']}")
        print()
    
    print("="*80)
    print(f"\n✅ Resultados guardados en 'financial_ratios_results.csv' y 'financial_health_analysis.csv'\n")
    
    return df_ratios, df_analysis


if __name__ == "__main__":
    df_ratios, df_analysis = main()

