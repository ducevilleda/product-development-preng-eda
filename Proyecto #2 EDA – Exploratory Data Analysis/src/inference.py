import os
import json
import pandas as pd
import joblib

from .utils import load_config


def _load_schema(schema_path: str) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_with_schema(X: pd.DataFrame, schema: dict) -> pd.DataFrame:
    # 1) columnas exactas y orden exacto
    cols = schema["feature_columns"]
    X = X.reindex(columns=cols)

    # 2) categorías EXACTAS 
    categories_map = schema.get("categories_map", {})
    for c, cats in categories_map.items():
        if c in X.columns:
            
            bad = ~X[c].isin(cats) & X[c].notna()
            if bad.any():
                bad_vals = X.loc[bad, c].unique().tolist()
                raise ValueError(f"Valor(es) no visto(s) en entrenamiento para '{c}': {bad_vals}")

            X[c] = pd.Categorical(X[c], categories=cats)

    # 3) NaNs: LightGBM 
    return X


def load_model_and_schema():
    config = load_config()
    model_path = config["output"]["model_path"]

    schema_path = config.get("output", {}).get(
        "schema_path",
        os.path.join(os.path.dirname(model_path), "model_schema.json")
    )

    model = joblib.load(model_path)
    schema = _load_schema(schema_path)
    return model, schema


def predict(data: dict):
    """
    data viene del API, ejemplo:
    {
      "store_nbr": 1,
      "family": "AUTOMOTIVE",
      "date": "2017-08-15",
      "onpromotion": 0
    }
    """

    # armar DataFrame base
    df = pd.DataFrame([data])

    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    

    model, schema = load_model_and_schema()

    X = align_with_schema(df, schema)

    yhat = model.predict(X)
    return float(yhat[0])
