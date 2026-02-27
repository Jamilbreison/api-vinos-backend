from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="API Predictor de Calidad del Vino")

# 1. Cargar los 3 archivos
modelo_lineal = joblib.load('modelo_lineal_vinos.pkl')
transformador_poly = joblib.load('transformador_poly.pkl')
modelo_poli = joblib.load('modelo_polinomial_vinos.pkl')

# 2. Definir la estructura de entrada
# Basado en el dataset tradicional de calidad de vino (11 variables)
class DatosVino(BaseModel):
    acidez_fija: float
    acidez_volatil: float
    acido_citrico: float
    azucar_residual: float
    cloruros: float
    dioxido_azufre_libre: float
    dioxido_azufre_total: float
    densidad: float
    pH: float
    sulfatos: float
    alcohol: float
    tipo_modelo: str = "lineal" # "lineal" o "polinomial"

@app.get("/")
def home():
    return {"mensaje": "API del Modelo de Vinos activa. Usa el endpoint /predecir."}

@app.post("/predecir")
def predecir_calidad(datos: DatosVino):
    # Crear un DataFrame con TODAS las variables en el orden correcto
    df_completo = pd.DataFrame([{
        'fixed acidity': datos.acidez_fija,
        'volatile acidity': datos.acidez_volatil,
        'citric acid': datos.acido_citrico,
        'residual sugar': datos.azucar_residual,
        'chlorides': datos.cloruros,
        'free sulfur dioxide': datos.dioxido_azufre_libre,
        'total sulfur dioxide': datos.dioxido_azufre_total,
        'density': datos.densidad,
        'pH': datos.pH,
        'sulphates': datos.sulfatos,
        'alcohol': datos.alcohol
    }])

    if datos.tipo_modelo == "lineal":
        # El modelo lineal usa las 11 variables
        prediccion = modelo_lineal.predict(df_completo)
    
    elif datos.tipo_modelo == "polinomial":
        # El modelo polinomial solo usa 3 (ajusta los nombres si usaste otros en Colab)
        df_reducido = df_completo[['alcohol', 'sulphates', 'citric acid']]
        
        # 1ro: Transformar los datos a grado 2
        datos_transformados = transformador_poly.transform(df_reducido)
        # 2do: Predecir con el modelo polinomial
        prediccion = modelo_poli.predict(datos_transformados)
        
    else:
        raise HTTPException(status_code=400, detail="Usa 'lineal' o 'polinomial'")

    return {
        "calidad_predicha": float(prediccion[0]),
        "modelo_utilizado": datos.tipo_modelo
    }
