import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Clase que maneja la predicción
class Prediccion:
    def __init__(self, modelo_path, scaler):
        # Cargar el modelo entrenado
        self.modelo = load_model(modelo_path)
        self.scaler = scaler

    def predecir(self, nuevo_individuo):
        # Normalizar las características
        nuevo_individuo_scaled = self.scaler.transform(nuevo_individuo[['Trip_Distance_km', 'Passenger_Count', 'Trip_Duration_Minutes']])

        # Concatenar las variables categóricas normalizadas
        nuevo_individuo_scaled = np.hstack((nuevo_individuo_scaled, nuevo_individuo[['Traffic_Conditions', 'Weather']].values))

        # Realizar la predicción con el modelo cargado
        prediccion = self.modelo.predict(nuevo_individuo_scaled)
        return prediccion[0][0]

# Crear un ejemplo de cómo se cargaría el modelo y hacer la predicción
# Primero, debes cargar el scaler previamente entrenado o crear uno nuevo si no lo guardaste.

# Cargar el dataset para ajustar el scaler
dataset = pd.read_csv("taxi_trip_pricing.csv")
X = dataset[['Trip_Distance_km', 'Passenger_Count', 'Trip_Duration_Minutes', 'Traffic_Conditions', 'Weather']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['Trip_Distance_km', 'Passenger_Count', 'Trip_Duration_Minutes']])

# Definir las características del nuevo individuo
nuevo_individuo = pd.DataFrame({
    'Trip_Distance_km': [3],
    'Traffic_Conditions': [1],
    'Passenger_Count': [4],
    'Weather': [2],
    'Trip_Duration_Minutes': [10]
})

# Crear la instancia de Prediccion usando el modelo cargado
prediccion_obj = Prediccion('modelo_entrenado.h5', scaler)

# Realizar la predicción
resultado = prediccion_obj.predecir(nuevo_individuo)

# Mostrar el resultado
print(f"Predicción del precio del viaje: $ {resultado}")


