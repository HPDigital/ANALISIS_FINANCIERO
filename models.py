"""
Modelos de predicción de precios de acciones.
Incluye: KNN, Random Forest, LSTM y Neural Network.
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Nota: imports de TensorFlow/Keras se realizan de forma diferida
# dentro de los métodos de las clases LSTM/NN para evitar
# demoras en el arranque de la aplicación al no usarse.
import pickle


class StockPredictionModel:
    """Clase base para modelos de predicción de acciones"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.scaler = None

    def train(self, X_train, y_train):
        """Entrena el modelo"""
        raise NotImplementedError

    def predict(self, X_test):
        """Realiza predicciones"""
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        """Calcula métricas de evaluación"""
        y_true = np.ravel(y_test)
        y_hat = np.ravel(y_pred)
        mae = mean_absolute_error(y_true, y_hat)
        mse = mean_squared_error(y_true, y_hat)
        rmse = np.sqrt(mse)
        return mae, mse, rmse

    def save_model(self, filepath):
        """Guarda el modelo"""
        raise NotImplementedError

    def load_model(self, filepath):
        """Carga el modelo"""
        raise NotImplementedError


class KNNModel(StockPredictionModel):
    """K-Nearest Neighbors para predicción de acciones"""

    def __init__(self, n_neighbors=5):
        super().__init__("K-Nearest Neighbors")
        self.n_neighbors = n_neighbors
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights='distance',
            algorithm='auto'
        )

    def train(self, X_train, y_train):
        """Entrena el modelo KNN"""
        print(f"\nEntrenando {self.model_name}...")
        self.model.fit(X_train, y_train)
        print(f"Modelo {self.model_name} entrenado exitosamente")
        return self

    def save_model(self, filepath):
        """Guarda el modelo KNN (incluye scaler si existe)"""
        bundle = {"model": self.model, "scaler": self.scaler}
        with open(filepath, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath):
        """Carga el modelo KNN (incluye scaler si fue guardado)"""
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
            if isinstance(obj, dict) and 'model' in obj:
                self.model = obj.get('model')
                self.scaler = obj.get('scaler')
            else:
                self.model = obj
                self.scaler = None
        print(f"Modelo cargado desde: {filepath}")
        return self


class RandomForestModel(StockPredictionModel):
    """Random Forest para predicción de acciones"""

    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        super().__init__("Random Forest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        """Entrena el modelo Random Forest"""
        print(f"\nEntrenando {self.model_name}...")
        self.model.fit(X_train, y_train)
        print(f"Modelo {self.model_name} entrenado exitosamente")
        return self

    def save_model(self, filepath):
        """Guarda el modelo Random Forest (incluye scaler si existe)"""
        bundle = {"model": self.model, "scaler": self.scaler}
        with open(filepath, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath):
        """Carga el modelo Random Forest (incluye scaler si fue guardado)"""
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
            if isinstance(obj, dict) and 'model' in obj:
                self.model = obj.get('model')
                self.scaler = obj.get('scaler')
            else:
                self.model = obj
                self.scaler = None
        print(f"Modelo cargado desde: {filepath}")
        return self


class LSTMModel(StockPredictionModel):
    """LSTM (Long Short-Term Memory) para series temporales"""

    def __init__(self, look_back=100, lstm_units=50, dropout_rate=0.2):
        super().__init__("LSTM")
        self.look_back = look_back
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self._build_model()

    def _build_model(self):
        """Construye la arquitectura del modelo LSTM"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
        self.model = Sequential([
            Input(shape=(self.look_back, 1)),
            LSTM(self.lstm_units, return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """Entrena el modelo LSTM"""
        print(f"\nEntrenando {self.model_name}...")

        # Reshape para LSTM (samples, timesteps, features)
        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        # Early stopping para evitar overfitting
        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            X_train_reshaped,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )

        print(f"Modelo {self.model_name} entrenado exitosamente")
        return history

    def predict(self, X_test):
        """Realiza predicciones con LSTM"""
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        return self.model.predict(X_test_reshaped, verbose=0)

    def save_model(self, filepath):
        """Guarda el modelo LSTM"""
        self.model.save(filepath)
        print(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath):
        """Carga el modelo LSTM"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        print(f"Modelo cargado desde: {filepath}")
        return self


class NeuralNetworkModel(StockPredictionModel):
    """Red Neuronal Densa para predicción de acciones"""

    def __init__(self, input_dim=100):
        super().__init__("Neural Network")
        self.input_dim = input_dim
        self._build_model()

    def _build_model(self):
        """Construye la arquitectura de la red neuronal"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Input
        self.model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """Entrena la red neuronal"""
        print(f"\nEntrenando {self.model_name}...")

        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )

        print(f"Modelo {self.model_name} entrenado exitosamente")
        return history

    def save_model(self, filepath):
        """Guarda la red neuronal"""
        self.model.save(filepath)
        print(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath):
        """Carga la red neuronal"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        print(f"Modelo cargado desde: {filepath}")
        return self


def compare_models(models_results):
    """Imprime comparación de métricas de modelos."""
    print("\n" + "="*80)
    print("COMPARACION DE MODELOS")
    print("="*80)
    print(f"{'Modelo':<25} {'MAE':<15} {'MSE':<15} {'RMSE':<15}")
    print("-"*80)

    for model_name, (mae, mse, rmse) in models_results.items():
        print(f"{model_name:<25} {mae:<15.4f} {mse:<15.4f} {rmse:<15.4f}")

    print("="*80)

    best_model = min(models_results.items(), key=lambda x: x[1][0])  # Por MAE
    print(f"\nMEJOR MODELO: {best_model[0]} (MAE: {best_model[1][0]:.4f})")
    print("="*80 + "\n")
