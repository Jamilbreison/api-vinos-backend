from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="API Predictor de Calidad del Vino")

# 1. Cargar los modelos
# NOTA: Asegúrate de que este modelo lineal se haya entrenado SOLO con las 3 variables.
# Si en Colab lo entrenaste con todo el dataset por error, debes reentrenarlo usando el CSV filtrado.
modelo_lineal = joblib.load('modelo_lineal_vinos.pkl')
transformador_poly = joblib.load('transformador_poly.pkl')
modelo_poli = joblib.load('modelo_polinomial_vinos.pkl')

# 2. Definir la estructura de entrada usando solo TUS 3 variables
class DatosVino(BaseModel):
    alcohol: float
    sulfato: float
    acido_citrico: float
    tipo_modelo: str = "lineal" # Puede ser "lineal" o "polinomial"

@app.get("/")
def home():
    return {"mensaje": "API del Modelo de Vinos activa. Usa el endpoint /predecir."}

@app.post("/predecir")
def predecir_calidad(datos: DatosVino):
    # 3. Crear el DataFrame EXACTAMENTE con los nombres de columna de tu CSV
    df_entrada = pd.DataFrame([{
        'alcohol': datos.alcohol,
        'sulfato': datos.sulfato,
        'acido citrico': datos.acido_citrico
    }])

    if datos.tipo_modelo == "lineal":
        prediccion = modelo_lineal.predict(df_entrada)
    
    elif datos.tipo_modelo == "polinomial":
        # Primero transformamos las 3 variables y luego predecimos
        datos_transformados = transformador_poly.transform(df_entrada)
        prediccion = modelo_poli.predict(datos_transformados)
        
    else:
        raise HTTPException(status_code=400, detail="Usa 'lineal' o 'polinomial'")

    return {
        "calidad_predicha": float(prediccion[0]),
        "modelo_utilizado": datos.tipo_modelo
    }
