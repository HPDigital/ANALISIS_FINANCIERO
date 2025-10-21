"""
Script para ejecutar todos los análisis financieros
Ejecuta predicción, ranking de portfolio y análisis de ratios
"""

import os
import sys
from datetime import datetime


def print_header(title):
    """Imprime un encabezado formateado"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def main():
    """Ejecuta todos los scripts del proyecto"""
    
    print_header("🚀 ANÁLISIS FINANCIERO COMPLETO")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Crear directorios necesarios
    os.makedirs('models_saved', exist_ok=True)
    os.makedirs('graficos', exist_ok=True)
    
    # 1. Predicción de Precios
    print_header("1️⃣ PREDICCIÓN DE PRECIOS DE ACCIONES")
    print("Ejecutando stock_prediction.py...\n")
    
    try:
        import stock_prediction
        print("\n✅ Predicción de precios completada")
    except Exception as e:
        print(f"\n❌ Error en predicción de precios: {str(e)}")
    
    # 2. Ranking de Portfolio
    print_header("2️⃣ RANKING DE PORTFOLIO (MAGIC FORMULA)")
    print("Ejecutando portfolio_ranking.py...\n")
    
    try:
        import portfolio_ranking
        print("\n✅ Ranking de portfolio completado")
    except Exception as e:
        print(f"\n❌ Error en ranking de portfolio: {str(e)}")
    
    # 3. Análisis de Ratios Financieros
    print_header("3️⃣ ANÁLISIS DE RATIOS FINANCIEROS")
    print("Ejecutando financial_ratios.py...\n")
    
    try:
        import financial_ratios
        print("\n✅ Análisis de ratios completado")
    except Exception as e:
        print(f"\n❌ Error en análisis de ratios: {str(e)}")
    
    # Resumen Final
    print_header("✨ RESUMEN FINAL")
    
    print("📁 Archivos generados:")
    print("  ✓ Modelos entrenados en models_saved/")
    print("  ✓ Gráficos interactivos en graficos/")
    print("  ✓ portfolio_ranking_results.csv")
    print("  ✓ financial_ratios_results.csv")
    print("  ✓ financial_health_analysis.csv")
    
    print(f"\n⏱️ Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("  🎉 ANÁLISIS COMPLETO FINALIZADO")
    print("="*80 + "\n")
    
    print("💡 Próximos pasos:")
    print("  1. Revisa los archivos CSV generados")
    print("  2. Explora los modelos guardados en models_saved/")
    print("  3. Ejecuta 'streamlit run app_streamlit.py' para la app web")
    print()


if __name__ == "__main__":
    main()

