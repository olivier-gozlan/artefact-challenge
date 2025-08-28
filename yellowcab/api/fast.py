from fastapi import FastAPI
import pandas as pd
import loguru
from pydantic import BaseModel
from yellowcab.model import instantiate_forest_model
from yellowcab.data import train_path,load_data, preprocess_data,compute_target,filter_outliers,encode_categorical_cols,extract_x_y
from yellowcab.train import predict_duration
from yellowcab.registry import load_model
from datetime import datetime
from fastapi import File,UploadFile
from typing import Annotated
import csv

class YellowCabFeatures(BaseModel):
    tpep_pickup_datetime: datetime
    tpep_dropoff_datetime:datetime
    PULocationID:int
    DOLocationID:int
    passenger_count:int
    
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
    return predict_actions(data)[0]
     
def predict_actions(data)->float:
    data_preprocess=preprocess_data(data)
    X,y,dv2=extract_x_y(data_preprocess,dv=dv)
    durations = predict_duration(X,model)
    return durations
        
 
@api.post("/predict")
def predict_post(features:YellowCabFeatures)->float:
    data = pd.DataFrame([features.model_dump()])
    return predict_actions(data)[0]


@api.post("/predict_batch")
def predict_batch(file: UploadFile = File(...)):
    logger.info("recup csv")
    # Définir les colonnes à parser en datetime
    date_cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]

# Définir les types pour les autres colonnes
    dtype_cols = [ "PULocationID","DOLocationID","passenger_count"]

# Charger le CSV
    
    csvReader = csv.DictReader(file.file.read().decode('utf-8').splitlines())
    logger.info("create dataframe")
    data = pd.DataFrame(csvReader)
    #data["tpep_pickup_datetime"]=pd.to_datetime(data["tpep_pickup_datetime"])
    #data["tpep_dropoff_datetime"]=pd.to_datetime(data["tpep_dropoff_datetime"])
    data[date_cols]=data[date_cols].apply(pd.to_datetime)
    data[dtype_cols]= data[dtype_cols].astype(int)
    print(data.dtypes)
    logger.info("predict")
    print(predict_actions(data).tolist())
    return predict_actions(data).tolist()