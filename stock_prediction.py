"""
Script principal para predicción de precios de acciones
Entrena y evalúa múltiples modelos de Machine Learning
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils import (
    download_stock_data,
    calculate_moving_averages,
    split_train_test,
    scale_data,
    create_dataset,
    plot_price_vs_ma,
    plot_predictions,
    print_metrics
)

from models import (
    KNNModel,
    RandomForestModel,
    LSTMModel,
    NeuralNetworkModel,
    compare_models
)


def main():
    """Función principal para ejecutar el análisis de predicción"""
    
    # ==================== CONFIGURACIÓN ====================
    print("\n" + "="*80)
    print("📈 PREDICCIÓN DE PRECIOS DE ACCIONES - MICROSOFT (MSFT)")
    print("="*80 + "\n")
    
    STOCK_SYMBOL = 'MSFT'
    START_DATE = '2000-01-01'
    END_DATE = '2024-02-01'
    LOOK_BACK = 100
    TRAIN_RATIO = 0.8
    
    # ==================== DESCARGA DE DATOS ====================
    data = download_stock_data(STOCK_SYMBOL, START_DATE, END_DATE)
    
    # ==================== MEDIAS MÓVILES ====================
    print("\n📊 Calculando medias móviles...")
    data = calculate_moving_averages(data, windows=[100, 200])
    
    # ==================== DIVISIÓN DE DATOS ====================
    print("\n🔀 Dividiendo datos en entrenamiento y prueba...")
    train_data, test_data = split_train_test(data, train_ratio=TRAIN_RATIO)
    
    # ==================== VISUALIZACIÓN ====================
    print("\n📊 Generando visualizaciones...")
    
    # Gráfico: Precio vs MA100
    plot_price_vs_ma(train_data, STOCK_SYMBOL, ma_windows=[100])
    
    # Gráfico: Precio vs MA100 vs MA200
    plot_price_vs_ma(train_data, STOCK_SYMBOL, ma_windows=[100, 200])
    
    # ==================== ESCALADO DE DATOS ====================
    print("\n🔄 Escalando datos...")
    train_scaled, test_scaled, scaler = scale_data(train_data, test_data)
    
    # ==================== PREPARACIÓN DE DATASETS ====================
    print(f"\n🔧 Creando datasets con look_back={LOOK_BACK}...")
    X_train, y_train = create_dataset(train_scaled, look_back=LOOK_BACK)
    X_test, y_test = create_dataset(test_scaled, look_back=LOOK_BACK)
    
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape y_train: {y_train.shape}")
    print(f"Shape X_test: {X_test.shape}")
    print(f"Shape y_test: {y_test.shape}")
    
    # ==================== ENTRENAMIENTO DE MODELOS ====================
    
    models_results = {}
    
    # --------------- 1. K-NEAREST NEIGHBORS (KNN) ---------------
    print("\n" + "="*80)
    print("1️⃣ MODELO: K-NEAREST NEIGHBORS (KNN)")
    print("="*80)
    
    knn_model = KNNModel(n_neighbors=5)
    knn_model.train(X_train, y_train)
    knn_model.save_model('models_saved/knn_model.pkl')
    
    # Predicciones
    knn_predictions = knn_model.predict(X_test)
    knn_predictions = scaler.inverse_transform(knn_predictions.reshape(-1, 1))
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Evaluación
    knn_mae, knn_mse, knn_rmse = knn_model.evaluate(y_test_inverse, knn_predictions)
    print_metrics(knn_mae, knn_mse, knn_rmse, "KNN")
    models_results['KNN'] = (knn_mae, knn_mse, knn_rmse)
    
    # Visualización
    plot_predictions(test_data, y_test_inverse, knn_predictions, "KNN", LOOK_BACK)
    
    # --------------- 2. RANDOM FOREST ---------------
    print("\n" + "="*80)
    print("2️⃣ MODELO: RANDOM FOREST")
    print("="*80)
    
    rf_model = RandomForestModel(n_estimators=100, max_depth=10)
    rf_model.train(X_train, y_train)
    rf_model.save_model('models_saved/random_forest_model.pkl')
    
    # Predicciones
    rf_predictions = rf_model.predict(X_test)
    rf_predictions = scaler.inverse_transform(rf_predictions.reshape(-1, 1))
    
    # Evaluación
    rf_mae, rf_mse, rf_rmse = rf_model.evaluate(y_test_inverse, rf_predictions)
    print_metrics(rf_mae, rf_mse, rf_rmse, "Random Forest")
    models_results['Random Forest'] = (rf_mae, rf_mse, rf_rmse)
    
    # Visualización
    plot_predictions(test_data, y_test_inverse, rf_predictions, "Random Forest", LOOK_BACK)
    
    # --------------- 3. LSTM (Long Short-Term Memory) ---------------
    print("\n" + "="*80)
    print("3️⃣ MODELO: LSTM (Long Short-Term Memory)")
    print("="*80)
    
    lstm_model = LSTMModel(look_back=LOOK_BACK, lstm_units=32, dropout_rate=0.2)
    lstm_model.train(X_train, y_train, epochs=20, batch_size=64)
    lstm_model.save_model('models_saved/lstm_model.keras')
    
    # Predicciones
    lstm_predictions = lstm_model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1, 1))
    
    # Evaluación
    lstm_mae, lstm_mse, lstm_rmse = lstm_model.evaluate(y_test_inverse, lstm_predictions)
    print_metrics(lstm_mae, lstm_mse, lstm_rmse, "LSTM")
    models_results['LSTM'] = (lstm_mae, lstm_mse, lstm_rmse)
    
    # Visualización
    plot_predictions(test_data, y_test_inverse, lstm_predictions, "LSTM", LOOK_BACK)
    
    # --------------- 4. NEURAL NETWORK (Dense) ---------------
    print("\n" + "="*80)
    print("4️⃣ MODELO: NEURAL NETWORK (Dense)")
    print("="*80)
    
    nn_model = NeuralNetworkModel(input_dim=LOOK_BACK)
    nn_model.train(X_train, y_train, epochs=20, batch_size=64)
    nn_model.save_model('models_saved/neural_network_model.keras')
    
    # Predicciones
    nn_predictions = nn_model.predict(X_test)
    nn_predictions = scaler.inverse_transform(nn_predictions.reshape(-1, 1))
    
    # Evaluación
    nn_mae, nn_mse, nn_rmse = nn_model.evaluate(y_test_inverse, nn_predictions)
    print_metrics(nn_mae, nn_mse, nn_rmse, "Neural Network")
    models_results['Neural Network'] = (nn_mae, nn_mse, nn_rmse)
    
    # Visualización
    plot_predictions(test_data, y_test_inverse, nn_predictions, "Neural Network", LOOK_BACK)
    
    # ==================== COMPARACIÓN FINAL ====================
    compare_models(models_results)
    
    print("\n✅ Proceso completado exitosamente!")
    print("📁 Modelos guardados en la carpeta 'models_saved/'\n")


if __name__ == "__main__":
    # Crear directorios necesarios
    import os
    os.makedirs('models_saved', exist_ok=True)
    os.makedirs('graficos', exist_ok=True)
    
    # Ejecutar análisis
    main()

