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

st.set_page_config(page_title="Modelo SVR", page_icon="📊")

st.markdown("# 📊 Modelo SVR")
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


""" 
Los comentarios a continuacion solo son una guia del formato del codigo, 
si no cumple con el modelo SVR eliminarlos y reemplazarlos por el codigo correspondiente
"""

# Transformacion de los datos


# Preprocesamiento de los datos



# Dividir los datos en conjunto de entrenamiento y prueba


# Escalar los datos


# Construccion del modelo SVR

# Compilar y entrenar el modelo



# ---> Descomentar el codigo de abajo con "##" al inicio cuando se creen las variables x_test,y_test,x_train y y_train

# Predecir la tendencia para el periodo de TEST, incluido el día siguiente

## test_predict = model.predict(x_test)

st.write("#### Dataframe con las predicciones")
# Crear DataFrame con las predicciones
## predictions_df = pd.DataFrame({'Valor Real': y_test, 'Predicciones': np.array(test_predict).flatten()})

# Mostrar el DataFrame con las predicciones

##st.write(predictions_df)

# Graficar precios reales y predicciones
## st.write("#### Grafico Precio Real vs Predicciones")
## plt.figure(figsize=(10, 6))
## plt.plot(y_test, color = 'red', label = 'Real Stock Price')
## plt.plot(test_predict, color = 'blue', label = 'Predicted Stock Price')
## plt.xlabel('Time')
## plt.ylabel('Tesla Stock Price')
## plt.title('Precios reales y predicciones')
## plt.legend()
## st.pyplot(plt)

with st.sidebar:
    st.write("🔼 Seleccione el modelo que desea ejecutar de la lista superior")