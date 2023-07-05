import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

#pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S') # Se capta la fecha y hora actual

#Importamos la librería nueva
import yfinance as yf

st.set_page_config(page_title="Modelo SVC", page_icon="🔍")

st.markdown("# 🔍 Modelo SVC")
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


# Definir la tendencia positiva
df.reset_index(inplace=True)
df = df.dropna()
df['Trend'] = np.where(df['Close'].shift(-1) > df['Close'],1,0)
#filas_con_null = df[df == 'null'].dropna(how='all')

#print(filas_con_null)
df = df.dropna(how='any')

# Dividir los datos en características (X) y etiquetas (y)

X = df.drop(columns=['Date','Trend'])  # Eliminar las columnas 'Date' y 'Trend' de las características
y = df['Trend']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear una instancia del modelo SVC

modelo = SVC()
modelo.fit(X_train, y_train)

# Predecir la tendencia en el conjunto de prueba
df['Predicted_Signal'] = modelo.predict(X)

st.write("#### Dataframe con las predicciones")
# Crear DataFrame con las predicciones
predictions_df = pd.DataFrame({'Tendencia Real': df['Trend'] , 'Tendencia predecida': df['Predicted_Signal'] })

# Mostrar el DataFrame con las predicciones
st.write(predictions_df)

fig = plt.figure()
plt.plot(df['Trend'][:50],color='red')
plt.plot(df['Predicted_Signal'][:50],color='blue')
st.pyplot(fig)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

accuracy = accuracy_score(df["Trend"], df["Predicted_Signal"])
st.write("#### Accuracy del modelo: ", accuracy)

conf_matrix = confusion_matrix(df["Trend"], df["Predicted_Signal"])
st.write("#### Matriz de confusión del modelo: ", conf_matrix)

# Calcular sensibilidad (recall)
recall = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
st.write("Sensibilidad del modelo:", recall)

# Calcular especificidad
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
st.write("Especificidad del modelo:", specificity)

# Calcular puntaje F1
f1_score = 2 * (recall * specificity) / (recall + specificity)
st.write("Puntaje F1 del modelo:", f1_score)

with st.sidebar:
    st.write("🔼 Seleccione el modelo que desea ejecutar de la lista superior")