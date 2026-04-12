import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestRegressor
import joblib
from dotenv import load_dotenv
import os

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

MODEL_PATH = "model.pkl"


def load_data():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT fecha, tmed, tmin, tmax, prec
    FROM public.aemet_observaciones
    ORDER BY fecha ASC;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def preprocess(df):

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


def train(df):

    features = [
        "tmed",
        "tmed_lag1",
        "tmed_lag2",
        "tmed_lag3",
        "tmed_lag7",
        "tmin",
        "tmax",
        "prec",
        "month",
        "dayofweek",
        "dayofyear"
    ]

    X = df[features]
    y = df["tmed"].shift(-1)

    X = X.iloc[:-1]
    y = y.dropna()

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model


def main():
    df = load_data()
    df = preprocess(df)
    model = train(df)
    joblib.dump(model, MODEL_PATH)
    print("Modelo guardado correctamente")


if __name__ == "__main__":
    main()