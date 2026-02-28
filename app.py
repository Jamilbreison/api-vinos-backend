import streamlit as st
import requests

st.title("Predicción de Calidad del Vino 🍷")
st.write("Ingresa las características del vino para predecir su calidad.")

# Entradas de usuario
alcohol = st.number_input("Nivel de Alcohol", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
sulfato = st.number_input("Nivel de Sulfato", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
acido_citrico = st.number_input("Ácido Cítrico", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

# Selección de modelo
modelo_seleccionado = st.selectbox("Selecciona el modelo de IA:", ["lineal", "polinomial"])

# IMPORTANTE: Reemplaza esta URL con la que te dé Render una vez publicado tu Backend
URL_API = "https://api-vinos-backend.onrender.com" 

if st.button("Obtener Predicción"):
    # Respetamos el espacio en 'acido citrico' para el JSON
    payload = {
        "alcohol": alcohol,
        "sulfato": sulfato,
        "acido citrico": acido_citrico
    }
    
    endpoint = f"{URL_API}/predecir/{modelo_seleccionado}"
    
    try:
        respuesta = requests.post(endpoint, json=payload)
        if respuesta.status_code == 200:
            resultado = respuesta.json()["prediccion_calidad"]
            st.success(f"La calidad estimada del vino es: **{resultado:.2f}**")
        else:
            st.error(f"Error en la API: {respuesta.text}")
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con el Backend. Verifica que la API esté en línea y la URL sea correcta.")
