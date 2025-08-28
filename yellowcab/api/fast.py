from fastapi import FastAPI
import pandas as pd
import loguru
from pydantic import BaseModel
from yellowcab.model import instantiate_forest_model
from yellowcab.data import train_path,load_data, preprocess_data,compute_target,filter_outliers,encode_categorical_cols,extract_x_y
from yellowcab.train import predict_duration
from datetime import datetime
from yellowcab.registry import load_model
model = instantiate_forest_model(fit=False)
logger = loguru.logger
api = FastAPI()

dv=load_model("vectorizer")

@api.get("/")
def read_root():
    return {"Hello": "World"}

@api.get("/predict")
def predict(tpep_pickup_datetime: datetime,tpep_dropoff_datetime:datetime,
            PULocationID:int, DOLocationID:int, passenger_count:int)->float:
    
  
    data = pd.DataFrame({
        "tpep_pickup_datetime": [tpep_pickup_datetime],
        "tpep_dropoff_datetime": [tpep_dropoff_datetime],
        "PULocationID": [PULocationID],
        "DOLocationID": [DOLocationID],
        "passenger_count": [passenger_count]
    })
    # Créer un DataFrame vide avec les mêmes colonnes et types
  
    logger.info(f"{data}")
    logger.info("before preprocess")
    
    
    data_preprocess=preprocess_data(data)
    print(data_preprocess)
    logger.info("preprocess ok, predict")
    X,y,dv2=extract_x_y(data_preprocess,dv=dv)
    print(X.shape)
    duration = predict_duration(X,model)
    print(duration)
    return float(duration[0])
    
 