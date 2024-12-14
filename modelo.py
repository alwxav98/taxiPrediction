import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Clase que maneja el entrenamiento del modelo
class Modelo:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None
        self.scaler = None

    def entrenar(self):
        # Preprocesamiento de los datos
        dataset = self.dataset

        # Verificar y corregir tipos de datos
        print("Tipos de datos antes de la limpieza:")
        print(dataset.dtypes)

        dataset['Traffic_Conditions'] = pd.factorize(dataset['Traffic_Conditions'])[0]
        dataset['Weather'] = pd.factorize(dataset['Weather'])[0]
        dataset['Trip_Price'] = pd.to_numeric(dataset['Trip_Price'], errors='coerce')

        # Eliminar filas con valores faltantes
        dataset = dataset.dropna()

        print("Tipos de datos después de la limpieza:")
        print(dataset.dtypes)

        # Dividir el dataset
        X = dataset[['Trip_Distance_km', 'Passenger_Count', 'Trip_Duration_Minutes', 'Traffic_Conditions', 'Weather']]
        y = dataset['Trip_Price']

        # Normalizar los datos
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X[['Trip_Distance_km', 'Passenger_Count', 'Trip_Duration_Minutes']])

        # Concatenar las variables categóricas normalizadas
        X_scaled = np.hstack((X_scaled, X[['Traffic_Conditions', 'Weather']].values))

        # Definir el modelo de red neuronal
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_dim=X_scaled.shape[1]),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Compilar el modelo
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

        # Entrenar el modelo
        self.model.fit(X_scaled, y, epochs=100, batch_size=32, validation_split=0.2)

        # Guardar el modelo entrenado
        self.model.save('modelo_entrenado.h5')
        print("Modelo guardado exitosamente.")

# Cargar el dataset y crear la instancia de Modelo
dataset = pd.read_csv("taxi_trip_pricing.csv")
modelo_obj = Modelo(dataset)
modelo_obj.entrenar()