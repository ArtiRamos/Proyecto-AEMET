from fastapi import FastAPI, Query
import psycopg2
import pandas as pd
from predict import predict_next_day
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
import os
import requests
import json


load_dotenv()

HF_API_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small" 
headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

def parse_query_llm(question: str):
    prompt = f"""Responde SOLO en JSON válido.
    Formato: {{"tipo":"media|maxima|minima","fecha":"YYYY-MM-DD"}}
    Pregunta: {question}"""

    try:
        response = requests.post(
        HF_MODEL_URL,
        headers=headers,
        json={"inputs": prompt}
    )

        data = response.json()

        if "error" in data:
            return None
            
        output = data[0]["generated_text"]

        match = re.search(r"\{.*}", output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None

    except Exception:
        return None


app = FastAPI(
    title="API de Predicción de Temperatura",
    description="Predicción basada en datos históricos del Aeropuerto de Madrid",
    version="1.0"
)

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
    conn.close()

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["prec"] = df["prec"].fillna(0)
    df = df.groupby("fecha").mean(numeric_only=True).reset_index()

    df["month"] = df["fecha"].dt.month
    df["dayofweek"] = df["fecha"].dt.dayofweek
    df["dayofyear"] = df["fecha"].dt.dayofyear

    for lag in [1, 2, 3, 7]:
        df[f"tmed_lag{lag}"] = df["tmed"].shift(lag)

    df = df.dropna()

    return df



@app.get("/forecast")
def forecast(days: int = Query(..., description="Número de días a predecir")):

    df = get_data().copy()
    preds = []

    for _ in range(days):
        pred = float(predict_next_day(df))
        pred = round(pred, 2)
        preds.append(pred)

        new_row = df.iloc[-1].copy()
        new_row["fecha"] = new_row["fecha"] + pd.Timedelta(days=1)
        new_row["tmed"] = pred

        new_row["month"] = new_row["fecha"].month
        new_row["dayofweek"] = new_row["fecha"].dayofweek
        new_row["dayofyear"] = new_row["fecha"].dayofyear

        new_row["tmed_lag1"] = df.iloc[-1]["tmed"]
        new_row["tmed_lag2"] = df.iloc[-2]["tmed"]
        new_row["tmed_lag3"] = df.iloc[-3]["tmed"]
        new_row["tmed_lag7"] = df.iloc[-7]["tmed"]

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return {
        "location": "Aeropuerto de Madrid",
        "forecast_days": days,
        "temperatures": preds,
        "unit": "°C"}



@app.get("/ask")
def ask(q: str = Query(..., description="Ejemplo: ¿Qué temperatura hizo ayer?")):

    df = get_data()
    df["fecha"] = pd.to_datetime(df["fecha"])

    parsed = parse_query_llm(q)



    if parsed is None:
        q_lower = q.lower()

        if "max" in q_lower:
            tipo = "maxima"
        elif "min" in q_lower:
            tipo = "minima"
        else:
            tipo = "media"

        today = df["fecha"].max()

        if "antes de ayer" in q_lower:
            target_date = today - timedelta(days=2)
        elif "ayer" in q_lower:
            target_date = today - timedelta(days=1)
        else:
            match = re.search(r"\d{2}/\d{2}/\d{4}", q_lower)
            if match:
                target_date = datetime.strptime(match.group(), "%d/%m/%Y")
            else:
                return {
                    "respuesta": "No se ha podido interpretar la fecha. Use 'ayer', 'antes de ayer' o dd/mm/yyyy."}



    else:
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



    if tipo == "maxima":
        column = "tmax"
        tipo_texto = "máxima"
    elif tipo == "minima":
        column = "tmin"
        tipo_texto = "mínima"
    else:
        column = "tmed"
        tipo_texto = "media"



    result = df[df["fecha"].dt.date == target_date.date()]

    if result.empty:
        return {
            "respuesta": f"No hay datos disponibles para el {target_date.strftime('%d/%m/%Y')} en el Aeropuerto de Madrid."}

    value = round(float(result.iloc[0][column]), 2)

    return {
        "respuesta": f"La temperatura {tipo_texto} registrada el {target_date.strftime('%d/%m/%Y')} en el Aeropuerto de Madrid fue de {value} °C."}