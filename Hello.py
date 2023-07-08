import streamlit as st

st.set_page_config(page_title="Modelos de predicción del Grupo A", page_icon=":smiley:")
st.title("📌Despliegue de modelos de Grupo A Inteligencia de Negocios📈💻")
with st.sidebar:
    st.write("🔼 Seleccione el modelo que desea ejecutar de la lista superior")
st.markdown(
    """
    ## Integrantes del Equipo A:
    | Nombre | Codigo|
    |--|--|
    | Laos Carrasco, Rafael Alonso |20200143|
    | Meza Ramón, Marco Alfredo Rubén |  |
    | Montes Pérez, Josue Justi |  |
    | Ortiz Urbai, Sebastian |20200056|
    | Suel Arroyo, Walter Nick | |
    | Tuyo Acero, Gustavo Alonso |20200302|
    | Velásquez Díaz, Ahmad Roshán |  |
    
    #

    ### Especificaciones: 

    Despliegue en la Web de sus modelos como un portal web que esté en línea: 
    
    Donde muestra las predicciones/los resultados:
    - Gráficamente. 
    - Númericamente los valores de las predicciones (print de dataframe con la predicción o clasificación).
    - De modo textual presentar una recomendación.
    
    Donde se muestra el EDA:

    - Ploteo de los precios reales. (Ploteo de media móvil los precios reales.)


    Donde el usuario pueda indicar:
    - El modelo ejecutar.
    - La acción o instrumento financiero que quiera analizar.
    - El rango de fechas.
"""
)
