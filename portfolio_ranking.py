"""
Portfolio Ranking usando la Magic Formula de Joel Greenblatt
Rankea acciones basándose en PE Ratio y Return on Assets (ROA)
"""

import pandas as pd
import yfinance as yf
from utils import get_stock_info, create_portfolio_dataframe


def calculate_magic_formula_ranking(stocks_list):
    """
    Calcula el ranking usando la Magic Formula
    
    Args:
        stocks_list (list): Lista de símbolos de acciones
        
    Returns:
        pd.DataFrame: DataFrame con rankings calculados
    """
    print("\n" + "="*80)
    print("📊 PORTFOLIO RANKING - MAGIC FORMULA")
    print("="*80 + "\n")
    
    # Obtener datos de todas las acciones
    df = create_portfolio_dataframe(stocks_list)
    
    if df.empty:
        print("❌ No se pudieron obtener datos de las acciones")
        return None
    
    # Calcular rankings
    print("\n🔢 Calculando rankings...")
    
    # Eliminar filas con datos faltantes en columnas clave
    df_clean = df.dropna(subset=['trailingPE', 'returnOnAssets']).copy()
    
    # PE Rank: Menor PE es mejor (ascending=True)
    df_clean['PE Rank'] = df_clean['trailingPE'].rank(ascending=True)
    
    # ROA Rank: Mayor ROA es mejor (ascending=False)
    df_clean['ROA Rank'] = df_clean['returnOnAssets'].rank(ascending=False)
    
    # Magic Rank: Suma de ambos rankings (menor es mejor)
    df_clean['Magic Rank'] = df_clean['PE Rank'] + df_clean['ROA Rank']
    
    df = df_clean
    
    # Seleccionar columnas relevantes
    df_selected = df.filter([
        'symbol',
        'shortName',
        'trailingPE',
        'returnOnAssets',
        'currentPrice',
        'marketCap',
        'PE Rank',
        'ROA Rank',
        'Magic Rank'
    ])
    
    # Ordenar por Magic Rank (menor es mejor)
    df_sorted = df_selected.sort_values(by='Magic Rank', ascending=True)
    
    return df_sorted


def display_portfolio_ranking(df_sorted):
    """
    Muestra el ranking del portfolio de forma organizada
    
    Args:
        df_sorted (pd.DataFrame): DataFrame con rankings ordenados
    """
    print("\n" + "="*80)
    print("🏆 RESULTADOS DEL RANKING")
    print("="*80 + "\n")
    
    for idx, row in df_sorted.iterrows():
        # Manejar valores NaN
        magic_rank = row['Magic Rank'] if pd.notna(row['Magic Rank']) else 999
        pe_rank = row['PE Rank'] if pd.notna(row['PE Rank']) else 999
        roa_rank = row['ROA Rank'] if pd.notna(row['ROA Rank']) else 999
        
        print(f"Rank #{int(magic_rank)}: {row['symbol']} - {row.get('shortName', 'N/A')}")
        print(f"  └─ PE Ratio: {row['trailingPE']:.2f} (Rank: {int(pe_rank)})" if pd.notna(row['trailingPE']) else "  └─ PE Ratio: N/A")
        print(f"  └─ ROA: {row['returnOnAssets']*100:.2f}% (Rank: {int(roa_rank)})" if pd.notna(row['returnOnAssets']) else "  └─ ROA: N/A")
        print(f"  └─ Precio: ${row.get('currentPrice', 0):.2f}" if row.get('currentPrice') else "  └─ Precio: N/A")
        print(f"  └─ Market Cap: ${row.get('marketCap', 0)/1e9:.2f}B" if row.get('marketCap') else "  └─ Market Cap: N/A")
        print()
    
    print("="*80)
    print(f"\n🥇 MEJOR ACCIÓN: {df_sorted.iloc[0]['symbol']} - {df_sorted.iloc[0].get('shortName', 'N/A')}")
    print("="*80 + "\n")


def main():
    """Función principal para ejecutar el ranking de portfolio"""
    
    # Lista de acciones a analizar (puedes agregar más)
    stocks = [
        'MO',      # Altria Group
        'AMN',     # AMN Healthcare Services
        'BTMD',    # Biote Corp
        'BKE',     # Buckle Inc
        'AAPL',    # Apple
        'MSFT',    # Microsoft
        'GOOGL',   # Alphabet
        'AMZN',    # Amazon
        'TSLA',    # Tesla
        'META',    # Meta
        'NVDA',    # Nvidia
        'JPM',     # JPMorgan Chase
        'V',       # Visa
        'WMT',     # Walmart
        'JNJ'      # Johnson & Johnson
    ]
    
    print(f"📋 Analizando {len(stocks)} acciones...")
    print(f"Símbolos: {', '.join(stocks)}\n")
    
    # Calcular ranking
    df_ranking = calculate_magic_formula_ranking(stocks)
    
    if df_ranking is not None:
        # Mostrar resultados
        display_portfolio_ranking(df_ranking)
        
        # Guardar resultados
        output_file = 'portfolio_ranking_results.csv'
        df_ranking.to_csv(output_file, index=False)
        print(f"💾 Resultados guardados en: {output_file}\n")
        
        return df_ranking
    
    return None


if __name__ == "__main__":
    df_results = main()
    
    # Mostrar tabla completa en pandas
    if df_results is not None:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print("\n📊 TABLA COMPLETA:")
        print(df_results.to_string(index=False))

