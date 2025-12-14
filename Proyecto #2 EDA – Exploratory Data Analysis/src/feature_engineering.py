import os
from typing import Tuple

import pandas as pd
import numpy as np

from .utils import load_config, ensure_dir


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga train.csv y test.csv desde la carpeta definida en config.yaml
    """
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    train_path = os.path.join(raw_dir, config["data"]["train_file"])
    test_path = os.path.join(raw_dir, config["data"]["test_file"])

    train = pd.read_csv(train_path, parse_dates=[config["keys"]["date_col"]])
    test = pd.read_csv(test_path, parse_dates=[config["keys"]["date_col"]])

    return train, test


def create_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Crea variables de calendario a partir de la fecha.
    """
    df["day_of_week"] = df[date_col].dt.weekday
    df["day"] = df[date_col].dt.day
    df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)
    df["month"] = df[date_col].dt.month
    df["year"] = df[date_col].dt.year
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
    return df


def create_lag_features(df: pd.DataFrame,
                        group_cols,
                        target_col: str) -> pd.DataFrame:
    """
    Crea lags y rolling windows por tienda+familia.
    """
    df = df.sort_values(group_cols + ["date"]).copy()

    # Lags
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)

    # Rolling means
    df["rolling_mean_7"] = (
        df.groupby(group_cols)[target_col]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    df["rolling_std_7"] = (
        df.groupby(group_cols)[target_col]
        .shift(1)
        .rolling(window=7)
        .std()
    )

    df["rolling_mean_28"] = (
        df.groupby(group_cols)[target_col]
        .shift(1)
        .rolling(window=28)
        .mean()
    )

    # Expanding mean (promedio histórico)
    df["expanding_mean"] = (
        df.groupby(group_cols)[target_col]
        .shift(1)
        .expanding()
        .mean()
    )

    return df


def make_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pipeline principal de Feature Engineering.
    Devuelve train, valid y test procesados.
    """
    config = load_config()
    date_col = config["keys"]["date_col"]
    store_col = config["keys"]["store_col"]
    family_col = config["keys"]["family_col"]
    target = config["target"]["name"]

    # 1. Cargar datos crudos
    train, test = load_raw_data()

    # Convertir family a categoría
    train[family_col] = train[family_col].astype("category")
    test[family_col] = test[family_col].astype("category")

    # 2. Features de tiempo
    train = create_time_features(train, date_col)
    test = create_time_features(test, date_col)

    # 3. Features de lags y rolling SOLO en train (test no tiene target)
    group_cols = [store_col, family_col]
    train = create_lag_features(train, group_cols, target)

    # 4. Eliminar filas con NaN generados por lags (primeras fechas)
    train = train.dropna().reset_index(drop=True)

    # 5. Split train / valid por fecha (último mes como valid)
    valid_months = config["time_series"]["valid_months"]
    max_date = train[date_col].max()
    split_date = max_date - pd.DateOffset(months=valid_months)

    train_df = train[train[date_col] <= split_date].reset_index(drop=True)
    valid_df = train[train[date_col] > split_date].reset_index(drop=True)

    # 6. Guardar en disco
    processed_dir = config["data"]["processed_dir"]
    ensure_dir(processed_dir)

    train_path = os.path.join(processed_dir, config["data"]["train_processed"])
    valid_path = os.path.join(processed_dir, config["data"]["valid_processed"])
    test_path = os.path.join(processed_dir, config["data"]["test_processed"])

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test.to_csv(test_path, index=False)

    return train_df, valid_df, test


if __name__ == "__main__":
    # Permite ejecutar: python -m src.feature_engineering
    make_features()
    print("✅ Feature engineering completado y datos guardados.")
