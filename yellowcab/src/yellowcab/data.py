import gdown
import os
import pandas as pd
import numpy as np
import loguru

from sklearn.feature_extraction import DictVectorizer

from typing import List
logger = loguru.logger

DATA_FOLDER = "./data"
train_path = f"{DATA_FOLDER}/yellow_tripdata_2021-01.parquet"
test_path = f"{DATA_FOLDER}/yellow_tripdata_2021-02.parquet"
predict_path = f"{DATA_FOLDER}/yellow_tripdata_2021-03.parquet"


def load_datas():
  if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    print(f"New directory {DATA_FOLDER} created!")

    gdown.download(
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-01.parquet",
        train_path,
        quiet=False,
    )
    gdown.download(
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-02.parquet",
        test_path,
        quiet=False,
    )
    gdown.download(
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-03.parquet",
        predict_path,
        quiet=False,
    )
  else :
    print("datas already exist")
    
def load_data(path: str):
    return pd.read_parquet(path)


def compute_target(
    df: pd.DataFrame,
    pickup_column: str = "tpep_pickup_datetime",
    dropoff_column: str = "tpep_dropoff_datetime",
) -> pd.DataFrame:
    df["duration"] = df[dropoff_column] - df[pickup_column]
    df["duration"] = df["duration"].dt.total_seconds() / 60
    return df


# MIN_DURATION = 1
# MAX_DURATION = 60
def filter_outliers(df: pd.DataFrame, min_duration: int = 1, max_duration: int = 60) -> pd.DataFrame:
    return df[df["duration"].between(min_duration, max_duration)]


# CATEGORICAL_COLS = ["PULocationID", "DOLocationID"]
def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = ["PULocationID", "DOLocationID", "passenger_count"]
    df[categorical_cols] = df[categorical_cols].fillna(-1).astype("int")
    df[categorical_cols] = df[categorical_cols].astype("str")
    return df


def extract_x_y(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    dv: DictVectorizer = None,
    with_target: bool = True,
) -> dict:
    logger.info(f"df dans extract:{df}")
    if categorical_cols is None:
        categorical_cols = ["PULocationID", "DOLocationID", "passenger_count"]
    dicts = df[categorical_cols].to_dict(orient="records")

    y = None
    if with_target:
        if dv is None:
            dv = DictVectorizer()
            dv.fit(dicts)
        y = df["duration"].values

    x = dv.transform(dicts)
    return x, y, dv

def preprocess_data(df):
    df = compute_target(df)
    df = filter_outliers(df)
    df = encode_categorical_cols(df)
    logger.info(f'preprocess_data return:{df}')
    return df
