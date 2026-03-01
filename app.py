import streamlit as st
import requests

st.title("Predicción de Calidad del Vino 🍷")
st.write("Ingresa las características del vino para predecir su calidad.")

# Entradas de usuario
alcohol = st.number_input("Nivel de Alcohol", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
sulfato = st.number_input("Nivel de Sulfato", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
acido_citrico = st.number_input("Ácido Cítrico", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

# Selección de modelo
modelo_seleccionado = st.selectbox("Selecciona el modelo de IA:", ["lineal Multiple", "polinomial"])

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
        # CAMBIO 1: Actualizamos el nombre esperado por la API
        if tipo_modelo == "lineal_multiple":
            prediccion = modelo_lineal.predict(df_entrada)[0]
            
        elif tipo_modelo == "polinomial":
            # Transformar los datos primero
            df_transformado = transformador.transform(df_entrada)
            prediccion = modelo_poli.predict(df_transformado)[0]
            
        else:
            # CAMBIO 2: Actualizamos el mensaje de error
            raise HTTPException(status_code=400, detail="Modelo no válido. Usa 'lineal_multiple' o 'polinomial'.")

        return {"prediccion_calidad": float(prediccion)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del modelo: {str(e)}")
