import os
import json
from typing import List, Tuple

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.lightgbm

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .utils import load_config, ensure_dir, set_seed
from .feature_engineering import make_features


def _get_mlflow_tracking_uri(config: dict) -> str:
    mlflow_cfg = config.get("mlflow", {})
    uri = (
        mlflow_cfg.get("tracking_uri")
        or mlflow_cfg.get("tracking_url")
        or mlflow_cfg.get("uri")
        or "file:./mlruns"
    )
    return str(uri)


def _prepare_mlflow_storage(tracking_uri: str) -> None:
    uri = tracking_uri.strip()

    if uri.lower().startswith("file:"):
        path = uri[5:]
        path = path.lstrip("/\\")
        if not path:
            path = "./mlruns"
        ensure_dir(path)
        return

    if uri.lower().startswith("sqlite"):
        parts = uri.split("sqlite:", 1)
        db_part = parts[1] if len(parts) > 1 else ""
        db_part = db_part.lstrip("/")

        
        if db_part.startswith("///"):
            db_path = db_part[3:]
        else:
            
            return

        db_dir = os.path.dirname(db_path)
        if db_dir:
            ensure_dir(db_dir)
        return


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config()
    processed_dir = config["data"]["processed_dir"]

    train_path = os.path.join(processed_dir, config["data"]["train_processed"])
    valid_path = os.path.join(processed_dir, config["data"]["valid_processed"])

    if not (os.path.exists(train_path) and os.path.exists(valid_path)):
        print("⚠️ Datos procesados no encontrados. Ejecutando Feature Engineering...")
        make_features()

    date_col = config["keys"]["date_col"]
    train_df = pd.read_csv(train_path, parse_dates=[date_col])
    valid_df = pd.read_csv(valid_path, parse_dates=[date_col])

    # Recuperar categorías al leer CSV 
    family_col = config["keys"].get("family_col")
    if family_col and family_col in train_df.columns:
        train_df[family_col] = train_df[family_col].astype("category")
        valid_df[family_col] = valid_df[family_col].astype("category")

    store_col = config["keys"].get("store_col")
    if store_col and store_col in train_df.columns:
        train_df[store_col] = train_df[store_col].astype("category")
        valid_df[store_col] = valid_df[store_col].astype("category")

    return train_df, valid_df


def get_feature_target(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=drop_cols)
    y = df[target_col].values
    return X, y


def _build_schema(X_train: pd.DataFrame) -> dict:
    """Guarda columnas y categorías EXACTAS usadas en entrenamiento."""
    feature_columns = list(X_train.columns)

    categorical_columns = []
    categories_map = {}

    for c in feature_columns:
        
        if pd.api.types.is_categorical_dtype(X_train[c]):
            categorical_columns.append(c)
            categories_map[c] = list(X_train[c].cat.categories)

    return {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "categories_map": categories_map,
    }


def train():
    config = load_config()
    set_seed(42)

    # ===== MLflow setup =====
    tracking_uri = _get_mlflow_tracking_uri(config)
    _prepare_mlflow_storage(tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config.get("mlflow", {}).get("experiment_name", "default")
    mlflow.set_experiment(experiment_name)

    # ===== 1) Cargar datos =====
    train_df, valid_df = load_processed_data()

    date_col = config["keys"]["date_col"]
    target_col = config["target"]["name"]

    # ===== 2) Separar X, y =====
    drop_cols = [target_col, date_col]
    X_train, y_train = get_feature_target(train_df, target_col, drop_cols)
    X_valid, y_valid = get_feature_target(valid_df, target_col, drop_cols)

    # ===== 3) Convertir object -> category =====
    for c in X_train.columns:
        if X_train[c].dtype == "object":
            X_train[c] = X_train[c].astype("category")
        if X_valid[c].dtype == "object":
            X_valid[c] = X_valid[c].astype("category")

    # alinear categorías del valid con las del train
    for c in X_train.columns:
        if pd.api.types.is_categorical_dtype(X_train[c]) and pd.api.types.is_categorical_dtype(X_valid[c]):
            X_valid[c] = X_valid[c].cat.set_categories(X_train[c].cat.categories)

    # ===== 4) Índices de columnas categóricas  =====
    categorical_features = config.get("categorical_features", [])
    cat_idxs = [X_train.columns.get_loc(c) for c in categorical_features if c in X_train.columns]

    # ===== 5) Modelo =====
    model_cfg = config["model"]
    params = model_cfg["params"]

    # rutas de salida
    output_path = config["output"]["model_path"]
    ensure_dir(os.path.dirname(output_path))

    schema_path = config.get(
        "output", {}
    ).get(
        "schema_path",
        os.path.join(os.path.dirname(output_path), "model_schema.json")
    )

    # guardar métricas en un json dentro de "models/"
    metrics_path = os.path.join(os.path.dirname(output_path), "metrics.json")

    with mlflow.start_run(run_name=model_cfg.get("name", "lgbm_run")):
        mlflow.log_params(params)

        model = LGBMRegressor(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            categorical_feature=cat_idxs if cat_idxs else "auto",
        )

        # ===== 6) Métricas =====
        y_pred = model.predict(X_valid)
        mse = mean_squared_error(y_valid, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_valid, y_pred))
        mape = float(np.mean(np.abs((y_valid - y_pred) / (y_valid + 1e-9))) * 100)

        print(f"✅ RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%")

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)

        # Guardar métricas localmente (en la misma carpeta del modelo)
        metrics = {"rmse": rmse, "mae": mae, "mape": mape}
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"📌 Métricas guardadas en: {metrics_path}")

        # ===== 7) Guardar modelo y schema =====
        
        try:
            mlflow.lightgbm.log_model(model, artifact_path="model")
        except Exception as e:
            print(f"⚠️ MLflow log_model falló (no detiene el entrenamiento): {e}")

        # Local
        joblib.dump(model, output_path)
        print(f"💾 Modelo guardado en: {output_path}")

        schema = _build_schema(X_train)
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
        print(f"🧾 Schema guardado en: {schema_path}")


if __name__ == "__main__":
    train()




