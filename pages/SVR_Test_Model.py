import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')

# Importamos la librería nueva
import yfinance as yf

st.set_page_config(page_title="Modelo SVR", page_icon="📊")

st.markdown("# 📊 Modelo SVR")
st.markdown("---")

# Set start and end dates for the price data
# Se establece la fecha de inicio y de fin para los datos de precios usando el input del usuario
st.write("Ingrese el rango de fechas para el análisis")
fechaInicio = st.date_input('Fecha de inicio', value=pd.to_datetime('2014-1-1'))
fechaFin = st.date_input('Fecha de fin', value=pd.to_datetime('today'))
st.markdown("---")
valor_txt = st.text_input("Ingrese el símbolo de valor a analizar", "BVN")


st.markdown("---")
df = yf.download(valor_txt, start=fechaInicio, end=fechaFin)
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

dfdates  = []
dfprices = []
# Preparing Data

###Use this function for pandas dataframes
def gd(dataframe):
    i = 1
    while i < len(dataframe):
        a = dataframe.iloc[i]
        #print(a)
        dfdates.append(i)
        #print(dates)
        dfprices.append(a[3])
        i += 1
    return

#input: dates, prices of length n for n days
#trains models, plots models, returns each model's prediction on day #x
def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

    svr_lin = SVR(kernel= 'linear', C= 1e3)
    svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
    print("Entrenando modelos (RBF)")
    svr_rbf.fit(dates, prices) # fitting the data points in the models
    print("Entrenando modelos (LIN)")
    svr_lin.fit(dates, prices)
    print("Entrenando modelos (POLY)")
    svr_poly.fit(dates, prices)

    plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints
    plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
    plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
    plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    st.pyplot(plt)
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

#parses in dataframe, creates dates and prices arrays, dates is just 1-n for calculation purposes
print("formateando datos")
df.reset_index(inplace=True)
gd(df)

#reverses prices list, which was fed in reverse chronological order originally
dfprices.reverse()

#trains models, so far model not fast enough for >20 day inputs
print("Prediciendo precios")
predicted_price = predict_prices(dfdates, dfprices, 20)

#prints out predictions from prediction function
print("Prediccion finalizada!")
st.write(predicted_price)

with st.sidebar:
    st.write("🔼 Seleccione el modelo que desea ejecutar de la lista superior")