from fastapi import FastAPI, Query
import psycopg2
import pandas as pd
from predict import predict_next_day


app = FastAPI(
    title="API de Temperatura - Aeropuerto de Madrid",
    description="Consulta y predicción de temperaturas basadas en datos históricos reales",
    version="1.0"
)


# CONEXIÓN DB

def get_data():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

    query = """
    SELECT fecha, tmed, tmin, tmax, prec
    FROM public.aemet_observaciones
    ORDER BY fecha ASC;
    """

    df = pd.read_sql(query, conn)

    df['fecha'] = pd.to_datetime(df['fecha'])
    df['prec'] = df['prec'].fillna(0)
    df = df.groupby('fecha').mean(numeric_only=True).reset_index()

    # features
    df['month'] = df['fecha'].dt.month
    df['dayofweek'] = df['fecha'].dt.dayofweek
    df['dayofyear'] = df['fecha'].dt.dayofyear

    for lag in [1, 2, 3, 7]:
        df[f'tmed_lag{lag}'] = df['tmed'].shift(lag)

    df = df.dropna()

    return df

#Se utiliza esta llamada a una api para 

import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

HF_API_TOKEN = os.getenv("HF_TOKEN")

def parse_query_llm(question: str):
    prompt = f"""Responde SOLO en JSON válido.
    Formato: {{"tipo":"media|maxima|minima","fecha":"YYYY-MM-DD"}}
    Convierte esta pregunta:{question}"""

    response = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-base",
        headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
        json={"inputs": prompt}
    )

    try:
        output = response.json()[0]["generated_text"]
        return json.loads(output)
    except:
        return None

    try:
        output = response.json()[0]["generated_text"]
        import json as js
        return js.loads(output)
    except:
        return None


# FORECAST (SOLO MAÑANA)

@app.get("/forecast")
def forecast():
    df = get_data()

    pred = float(predict_next_day(df))
    pred = round(pred, 2)

    return {
        "location": "Aeropuerto de Madrid",
        "type": "prediccion",
        "description": "Temperatura media estimada para mañana",
        "temperature": pred,
        "unit": "°C"
    }



# ASK (CONSULTA NATURAL)

@app.get("/ask")
def ask(q: str = Query(..., description="Ejemplo: ¿Qué temperatura hizo ayer?")):

    df = get_data()

    parsed = parse_query_llm(q)

    if not parsed:
        return {
            "respuesta": "No se ha podido interpretar la pregunta. Intente con una fecha clara (por ejemplo: 12/03/2024 o 'ayer')."}

    tipo = parsed.get("tipo", "media")
    fecha_str = parsed.get("fecha")

    if not fecha_str:
        return {
            "respuesta": "No se ha podido identificar la fecha en la pregunta."}

    try:
        target_date = pd.to_datetime(fecha_str)
    except:
        return {
            "respuesta": "El formato de fecha no es válido."}

    # Selección de columna

    if tipo == "maxima":
        column = "tmax"
        tipo_texto = "máxima"
    elif tipo == "minima":
        column = "tmin"
        tipo_texto = "mínima"
    else:
        column = "tmed"
        tipo_texto = "media"

    result = df[df["fecha"] == target_date]

    if result.empty:
        return {
            "respuesta": f"No hay datos disponibles para el {target_date.strftime('%d/%m/%Y')} en el Aeropuerto de Madrid"}

    value = round(float(result.iloc[0][column]), 2)

    return {
        "respuesta": f"La temperatura {tipo_texto} registrada el {target_date.strftime('%d/%m/%Y')} en el Aeropuerto de Madrid fue de {value} °C."}