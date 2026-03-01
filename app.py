import streamlit as st
import requests

st.title("Predicción de Calidad del Vino 🍷")
st.write("Ingresa las características del vino para predecir su calidad.")

# Entradas de usuario
alcohol = st.number_input("Nivel de Alcohol", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
sulfato = st.number_input("Nivel de Sulfato", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
acido_citrico = st.number_input("Ácido Cítrico", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

# Diccionario para "traducir" el nombre visual al nombre que espera la API
nombres_modelos = {
    "Lineal Múltiple": "lineal_multiple",
    "Polinomial": "polinomial"
}

# Selección de modelo (el usuario ve los nombres bonitos)
nombre_visual = st.selectbox("Selecciona el modelo de IA:", list(nombres_modelos.keys()))

# Internamente sacamos el nombre seguro para la URL (con guion bajo)
modelo_seleccionado = nombres_modelos[nombre_visual]

# IMPORTANTE: Esta es la URL de tu Backend en Render
URL_API = "https://api-vinos-backend.onrender.com"

if st.button("Obtener Predicción"):
    # Respetamos el espacio en 'acido citrico' para el JSON que espera FastAPI
    payload = {
        "alcohol": alcohol,
        "sulfato": sulfato,
        "acido citrico": acido_citrico
    }
    
    # Armamos la ruta exacta (ej. .../predecir/lineal_multiple)
    endpoint = f"{URL_API}/predecir/{modelo_seleccionado}"
    
    try:
        respuesta = requests.post(endpoint, json=payload)
        
        if respuesta.status_code == 200:
            resultado = respuesta.json()["prediccion_calidad"]
            st.success(f"La calidad estimada del vino es: **{resultado:.2f}**")
        else:
            # Si hay un error, mostrará el mensaje exacto que configuramos en FastAPI
            st.error(f"Error en la API: {respuesta.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con el Backend. Verifica que la API esté en línea y la URL sea correcta.")
