import streamlit as st
import matplotlib.pyplot as plt  # Importación de la biblioteca matplotlib.pyplot para graficar datos
import pandas as pd  # Importación de la biblioteca pandas para manipulación de datos en forma de DataFrame
import numpy as np  # Importación de la biblioteca numpy para operaciones numéricas

from datetime import datetime
import seaborn as sns

from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S') # Se capta la fecha y hora actual

#Importamos la librería nueva
import yfinance as yf

st.set_page_config(page_title="Modelo LSTM", page_icon="📈")

st.markdown("# 📈 Modelo LSTM")
st.markdown("---")

# Set start and end dates for the price data
# Se establece la fecha de inicio y de fin para los datos de precios usando el input del usuario
st.write("Ingrese el rango de fechas para el análisis")
fechaInicio = st.date_input('Fecha de inicio' , value=pd.to_datetime('2014-1-1'))  
fechaFin = st.date_input('Fecha de fin' , value=pd.to_datetime('today'))
st.markdown("---")
valor_txt = st.text_input("Ingrese el simbolo de valor a analizar", "BVN")


st.markdown("---")
df = yf.download(valor_txt, start = fechaInicio, end = fechaFin)
#Añadimos la terminación _BVN a cada columna y mostramos el dataframe
#df.columns += "_%s" % valor_txt
st.write("#### Visualizacion del dataframe obtenido de Yahoo Finance para el valor", valor_txt) 
st.write(df)
st.write("%i filas y %i columnas" % (df.shape[0], df.shape[1]))

st.markdown("---")

# Ploteamos la media móvil de 20 días
st.write("#### Gráfico de Precio Real (Columna Close) y media móvil de 20 días")
rolling_mean = df['Close'].rolling(window=20).mean()
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label='Precios reales')
plt.plot(df.index, rolling_mean, label='Media móvil (20 días)')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.title('Precios reales y media móvil')
plt.legend()
st.pyplot(plt)

# Creando funciones de transformacion
def transformarSetASep(fecha):
  return fecha.replace('Set', 'Sep')

def transformarFechaDiaria(fecha):
  fechaNueva = datetime.strptime(fecha, '%d.%b.%y')
  return fechaNueva

def transformarFechaMensual(fecha):
  fechaNueva = datetime.strptime(fecha, '%b.%Y')
  return fechaNueva

def transformarFechaConHora(fecha):
  fechaNueva = pd.to_datetime(fecha,
               format='%Y-%m-%d')
  return fechaNueva

# Transformacion de los datos
df.reset_index(inplace=True)
df['Date'] = df['Date'].apply(transformarFechaConHora)
df = df.set_index('Date')

# Preprocesamiento de los datos
df['Return'] = df['Close'].pct_change()
df['Return'] = df['Return'].fillna(0)
df = df.dropna(how='any')  # Eliminar filas con valores nulos

# Dividir los datos en conjunto de entrenamiento y prueba
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# Escalar los datos
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data.drop(['Return'], axis=1))
scaled_test_data = scaler.transform(test_data.drop(['Return'], axis=1))

# Crear secuencias de tiempo para el modelo LSTM
window_size = 100

def create_sequences(data):
    x = []
    y = []
    for i in range( window_size , len(data) ):

        x.append( data[i-window_size:i] )
        print("interacion: ", i-window_size +1 )
        print("indicador: ", i)
        print("posicion inicial tomada: ", i-window_size)
        print("posicion final tomada: ", i-1)

        print(data[i])
        # print(data[i-window_size:i])

        y.append( data[i][-1] )
        print("y[",i,"]: ", data[i][-1])


    return np.array(x), np.array(y)


with st.spinner('Espere mientras se crean las secuencias para el modelo LSTM...'):
    x_train, y_train = create_sequences(scaled_train_data)
    x_test, y_test = create_sequences(scaled_test_data)


# Construccion del modelo LSTM
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import adam

model = Sequential()
model.add(LSTM(units=160, return_sequences=True,
               input_shape=(x_train.shape[1], x_train.shape[2])))
# model.add(Dropout(0.2))
model.add(LSTM(units=160))
# model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

# Compilar el modelo
model.compile(
    optimizer = adam.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy'])

st.markdown("---")
with st.spinner('Espere mientras se realiza la predicción...'):
    model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=1)
st.success('Predicción realizada con éxito!')

# Predecir la tendencia para el periodo de TEST, incluido el día siguiente
test_predict = model.predict(x_test)


st.write("#### Dataframe con las predicciones")
# Crear DataFrame con las predicciones
predictions_df = pd.DataFrame({'Valor Real': y_test, 'Predicciones': np.array(test_predict).flatten()})

# Mostrar el DataFrame con las predicciones
st.write(predictions_df)

# Graficar precios reales y predicciones
st.write("#### Grafico Precio Real vs Predicciones")
plt.figure(figsize=(10, 6))
plt.plot(y_test, color = 'red', label = 'Real Stock Price')
plt.plot(test_predict, color = 'blue', label = 'Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Tesla Stock Price')
plt.title('Precios reales y predicciones')
plt.legend()
st.pyplot(plt)

with st.sidebar:
    st.write("🔼 Seleccione el modelo que desea ejecutar de la lista superior")