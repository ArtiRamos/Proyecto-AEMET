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
    q_lower = q.lower()

    last = df.iloc[-1]

    if "ayer" in q_lower:

        if "max" in q_lower:
            value = round(float(last['tmax']), 2)
            tipo = "máxima"

        elif "min" in q_lower:
            value = round(float(last['tmin']), 2)
            tipo = "mínima"

        else:
            value = round(float(last['tmed']), 2)
            tipo = "media"

        return {
            "location": "Aeropuerto de Madrid",
            "question": q,
            "temperature": value,
            "type": tipo,
            "unit": "°C"
        }

    return {
        "error": "Consulta no reconocida",
        "help": "Puedes preguntar: temperatura media, mínima o máxima de ayer"
    }