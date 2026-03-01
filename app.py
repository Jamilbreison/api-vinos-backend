import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predicción de Vinos", layout="wide")

st.title("Predicción de Calidad del Vino 🍷")
st.write("Ingresa las características del vino para predecir su calidad y explora el comportamiento del modelo.")

# --- DICCIONARIO DE MODELOS ---
nombres_modelos = {
    "Lineal Múltiple": "lineal_multiple",
    "Polinomial (Grado 2)": "polinomial"
}

# --- BARRA LATERAL (ENTRADAS DEL USUARIO) ---
st.sidebar.header("Parámetros del Vino")
alcohol = st.sidebar.number_input("Nivel de Alcohol", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
sulfato = st.sidebar.number_input("Nivel de Sulfato", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
acido_citrico = st.sidebar.number_input("Ácido Cítrico", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

nombre_visual = st.selectbox("Selecciona el modelo de IA:", list(nombres_modelos.keys()))
modelo_seleccionado = nombres_modelos[nombre_visual]

# --- 1. SECCIÓN DE PREDICCIÓN (API) ---
URL_API = "https://api-vinos-backend.onrender.com"

if st.button("Obtener Predicción Individual", type="primary"):
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
        st.error("No se pudo conectar con el Backend.")

st.divider()

# --- 2. SECCIÓN DE ANÁLISIS VISUAL ---
st.header(f"Análisis del Modelo: {nombre_visual}")

# Cargar datos y modelos en caché para que no ralentice la app
@st.cache_data
def cargar_datos():
    df = pd.read_csv('wine_quality_filtrado.csv')
    return df

@st.cache_resource
def cargar_modelos():
    lin = joblib.load('modelo_lineal_vinos.pkl')
    poly = joblib.load('modelo_polinomial_vinos.pkl')
    trans = joblib.load('transformador_poly.pkl')
    return lin, poly, trans

try:
    df = cargar_datos()
    modelo_lineal, modelo_poli, transformador = cargar_modelos()
    
    # Preparar datos (replicando tu código de Colab)
    X = df[['alcohol', 'sulfato', 'acido citrico']]
    y = df['calidad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Generar predicciones según el modelo seleccionado
    if modelo_seleccionado == "lineal_multiple":
        y_pred = modelo_lineal.predict(X_test)
        modelo_activo = modelo_lineal
    else: # polinomial
        y_pred = modelo_poli.predict(transformador.transform(X_test))
        modelo_activo = modelo_poli
        
    # Calcular Métricas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Mostrar Métricas en columnas
    col1, col2 = st.columns(2)
    col1.metric("R² (Varianza Explicada)", f"{r2:.4f}", f"{r2*100:.1f}%")
    col2.metric("RMSE (Error Promedio)", f"{rmse:.4f}")
    
    # --- GRÁFICAS 2D ---
    st.subheader("Visualizaciones 2D")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # Valores Reales vs Predichos
        fig1 = px.scatter(x=y_test, y=y_pred, opacity=0.6, labels={'x': 'Valores Reales', 'y': 'Predicciones'})
        fig1.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="red", dash="dash"))
        fig1.update_layout(title="Valores Reales vs Predichos")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_g2:
        # Residuos
        residuos = y_test - y_pred
        fig2 = px.scatter(x=y_pred, y=residuos, opacity=0.6, labels={'x': 'Predicciones', 'y': 'Residuos (Errores)'}, color_discrete_sequence=['purple'])
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        fig2.update_layout(title="Análisis de Residuos")
        st.plotly_chart(fig2, use_container_width=True)

    # --- GRÁFICA 3D (PLANO/SUPERFICIE) ---
    st.subheader("Superficie de Predicción 3D interactiva")
    st.write("*(El Ácido Cítrico se mantiene en su valor promedio para visualizar la superficie en 3 dimensiones. Puedes rotar y hacer zoom)*")
    
    # Crear la malla (meshgrid) para la superficie
    rango_alcohol = np.linspace(X['alcohol'].min(), X['alcohol'].max(), 20)
    rango_sulfato = np.linspace(X['sulfato'].min(), X['sulfato'].max(), 20)
    x_surf, y_surf = np.meshgrid(rango_alcohol, rango_sulfato)
    
    # Crear DataFrame temporal para predecir la superficie
    df_surf = pd.DataFrame({
        'alcohol': x_surf.ravel(),
        'sulfato': y_surf.ravel(),
        'acido citrico': X['acido citrico'].mean()
    })
    
    # Predecir los valores Z de la superficie
    if modelo_seleccionado == "lineal_multiple":
        z_surf = modelo_lineal.predict(df_surf).reshape(x_surf.shape)
    else:
        z_surf = modelo_poli.predict(transformador.transform(df_surf)).reshape(x_surf.shape)

    # Dibujar plano y puntos con Plotly
    fig3 = go.Figure()
    
    # Puntos reales (Dataset)
    fig3.add_trace(go.Scatter3d(
        x=X_test['alcohol'], y=X_test['sulfato'], z=y_test,
        mode='markers', marker=dict(size=4, color='orange', opacity=0.5), name='Datos Reales'
    ))
    
    # Superficie del modelo
    fig3.add_trace(go.Surface(
        x=x_surf, y=y_surf, z=z_surf, 
        colorscale='Viridis', opacity=0.7, name='Superficie Predicha', showscale=False
    ))
    
    fig3.update_layout(
        scene=dict(
            xaxis_title='Alcohol',
            yaxis_title='Sulfato',
            zaxis_title='Calidad'
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

except FileNotFoundError as e:
    st.error(f"Falta un archivo para generar las gráficas: {e}. Asegúrate de subir 'wine_quality_filtrado.csv' y tus archivos .pkl a GitHub.")
except Exception as e:
    st.error(f"Error al generar visualizaciones: {e}")
