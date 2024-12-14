import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
dataset = pd.read_csv("C:/Users/tanya/Downloads/taxi_trip_pricing.csv")

# Exploración inicial del dataset
print(dataset.head())
print(dataset.describe())
print(dataset.info())

# Preprocesamiento de datos
## Manejar valores faltantes
dataset = dataset.dropna()

## Eliminar outliers de 'Trip_Price'
q1 = dataset['Trip_Price'].quantile(0.25)
q3 = dataset['Trip_Price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
dataset = dataset[(dataset['Trip_Price'] >= lower_bound) & (dataset['Trip_Price'] <= upper_bound)]

# Convertir variables categóricas a variables dummy (one-hot encoding)
dataset['Traffic_Conditions'] = pd.factorize(dataset['Traffic_Conditions'])[0]
dataset['Weather'] = pd.factorize(dataset['Weather'])[0]

# Dividir el dataset en conjunto de entrenamiento y prueba
X = dataset[['Trip_Distance_km', 'Passenger_Count', 'Trip_Duration_Minutes', 'Traffic_Conditions', 'Weather']]
y = dataset['Trip_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['Trip_Distance_km', 'Passenger_Count', 'Trip_Duration_Minutes']])
X_test_scaled = scaler.transform(X_test[['Trip_Distance_km', 'Passenger_Count', 'Trip_Duration_Minutes']])

# Concatenar las variables categóricas normalizadas
X_train_scaled = np.hstack((X_train_scaled, X_train[['Traffic_Conditions', 'Weather']].values))
X_test_scaled = np.hstack((X_test_scaled, X_test[['Traffic_Conditions', 'Weather']].values))

# Definir el modelo de red neuronal
model_nn = Sequential([
    Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),  # Primera capa densa
    Dense(32, activation='relu'),
    Dense(1)  # Salida
])

# Compilar el modelo
model_nn.compile(loss='mean_squared_error',
                 optimizer=Adam(),
                 metrics=['mean_absolute_error'])

# Entrenar el modelo
history = model_nn.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluar el modelo en el conjunto de prueba
predictions_nn = model_nn.predict(X_test_scaled)
rmse_nn = np.sqrt(np.mean((predictions_nn - y_test.values.reshape(-1, 1))**2))


# Visualizar resultados
results_nn = pd.DataFrame({'Actual': y_test, 'Predicted': predictions_nn.flatten()})
plt.scatter(results_nn['Actual'], results_nn['Predicted'], color='blue')
plt.plot([results_nn['Actual'].min(), results_nn['Actual'].max()], [results_nn['Actual'].min(), results_nn['Actual'].max()], color='red')
plt.title('Actual vs Predicted Taxi Fares - Neural Network')
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.show()

# Definir las características del nuevo individuo
new_data_nn = pd.DataFrame({
    'Trip_Distance_km': [36.87],         # Distancia en kilómetros
    'Traffic_Conditions': [2],         # Condición del tráfico (baja)
    'Passenger_Count': [1],           # Número de pasajeros
    'Weather': [0],                   # Clima (despejado)
    'Trip_Duration_Minutes': [37.27]     # Duración del viaje en minutos
})

# Normalizar las características
new_data_nn_scaled = scaler.transform(new_data_nn[['Trip_Distance_km', 'Passenger_Count', 'Trip_Duration_Minutes']])

# Concatenar las variables categóricas normalizadas
new_data_nn_scaled = np.hstack((new_data_nn_scaled, new_data_nn[['Traffic_Conditions', 'Weather']].values))

# Realizar la predicción con la red neuronal entrenada
predicted_price_nn = model_nn.predict(new_data_nn_scaled)
print(f"Predicción del precio del viaje (Red Neuronal): {predicted_price_nn[0][0]}")


