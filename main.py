from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="API de Predicción de Vinos")

# Cargar los modelos y el transformador
try:
    modelo_lineal = joblib.load('modelo_lineal_vinos.pkl')
    modelo_poli = joblib.load('modelo_polinomial_vinos.pkl')
    transformador = joblib.load('transformador_poly.pkl')
except Exception as e:
    print(f"Error cargando modelos: {e}")

# Definir la estructura del JSON que recibirá la API
class DatosVino(BaseModel):
    alcohol: float
    sulfato: float
    # Usamos alias para aceptar el espacio en el JSON entrante
    acido_citrico: float = Field(alias="acido citrico")

@app.post("/predecir/{tipo_modelo}")
def predecir_calidad(tipo_modelo: str, datos: DatosVino):
    # Reconstruir el DataFrame respetando estrictamente los nombres de las columnas
    df_entrada = pd.DataFrame([{
        "alcohol": datos.alcohol,
        "sulfato": datos.sulfato,
        "acido citrico": datos.acido_citrico
    }])

    try:
        # AQUÍ ESTÁ EL CAMBIO: Ahora el backend espera "lineal_multiple"
        if tipo_modelo == "lineal_multiple":
            prediccion = modelo_lineal.predict(df_entrada)[0]
            
        elif tipo_modelo == "polinomial":
            # Transformar los datos primero
            df_transformado = transformador.transform(df_entrada)
            prediccion = modelo_poli.predict(df_transformado)[0]
            
        else:
            # También actualizamos el mensaje de error
            raise HTTPException(status_code=400, detail="Modelo no válido. Usa 'lineal_multiple' o 'polinomial'.")

        return {"prediccion_calidad": float(prediccion)}
        
    except Exception as e:
        # Enviamos el error exacto al Frontend si algo falla
        raise HTTPException(status_code=500, detail=f"Error interno del modelo: {str(e)}")
