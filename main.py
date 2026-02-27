from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="API Predictor de Calidad del Vino")

# Cargar los modelos directamente
modelo_lineal = joblib.load('modelo_lineal_base.pkl')
modelo_poli = joblib.load('modelo_polinomial_g2.pkl')

# Definir la estructura de entrada
class DatosVino(BaseModel):
    alcohol: float
    sulfato: float
    acido_citrico: float
    tipo_modelo: str = "lineal" # El usuario podrá enviar "lineal" o "polinomial"

@app.get("/")
def home():
    return {"mensaje": "API activa. Envía datos al endpoint /predecir"}

@app.post("/predecir")
def predecir_calidad(datos: DatosVino):
    # Seleccionar el modelo
    if datos.tipo_modelo == "lineal":
        modelo = modelo_lineal
    elif datos.tipo_modelo == "polinomial":
        modelo = modelo_poli
    else:
        raise HTTPException(status_code=400, detail="Modelo no soportado. Usa 'lineal' o 'polinomial'.")

    # Crear el DataFrame respetando los nombres de las columnas de entrenamiento
    df_entrada = pd.DataFrame([{
        'alcohol': datos.alcohol,
        'sulfato': datos.sulfato,
        'ácido cítrico': datos.acido_citrico 
    }])

    # La predicción funciona igual para ambos gracias al Pipeline del polinomial
    prediccion = modelo.predict(df_entrada)
    
    return {
        "calidad_predicha": float(prediccion[0]),
        "modelo_utilizado": datos.tipo_modelo
    }
