import joblib
import pandas as pd

model = joblib.load("model.pkl")

def predict_next_day(df: pd.DataFrame):

    last = df.iloc[-1]

    features = pd.DataFrame([{
        "tmed": last["tmed"],
        "tmed_lag1": last["tmed_lag1"],
        "tmed_lag2": last["tmed_lag2"],
        "tmed_lag3": last["tmed_lag3"],
        "tmed_lag7": last["tmed_lag7"],
        "tmin": last["tmin"],
        "tmax": last["tmax"],
        "prec": last["prec"],
        "month": last["month"],
        "dayofweek": last["dayofweek"],
        "dayofyear": last["dayofyear"]
    }])

    pred = model.predict(features)[0]

    return pred