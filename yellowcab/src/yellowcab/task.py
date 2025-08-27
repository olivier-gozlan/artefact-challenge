from prefect import get_run_logger
from prefect import task ,flow
import loguru
from yellowcab.model import create_linearreg_model, instantiate_forest_model
from yellowcab.data import preprocess_data, load_datas, load_data, compute_target, filter_outliers, encode_categorical_cols, extract_x_y
from yellowcab.train import train_model, predict_duration, evaluate_model
from yellowcab import data
 
logger = loguru.logger

@task
def load_data_task(*args,**kwargs):
    return load_data(*args,**kwargs)

@task
def preprocess_data_task(*args,**kwargs):
    return preprocess_data(*args,**kwargs)

@task    
def extract_x_y_task(*args,**kwargs):
    return extract_x_y(*args,**kwargs)

@task    
def instantiate_forest_model_task(*args,**kwargs):
    return instantiate_forest_model(*args,**kwargs)

@task
def train_model_task(*args,**kwargs):
    return train_model(*args,**kwargs)

@task
def predict_duration_task(*args,**kwargs):
    return predict_duration(*args,**kwargs)

@task
def evaluate_model_task(*args,**kwargs):
    return evaluate_model(*args,**kwargs)
   
   
@flow
def train():
    logger = get_run_logger()
    datas = load_data_task(data.train_path)
    data_preproc = preprocess_data_task(datas)
    X_train, y_train, dv = extract_x_y_task(data_preproc)
    logger.info("✅ Data preprocessing complete")  
    model = instantiate_forest_model_task(fit=True)
    train_model_task(model,X_train,y_train)
    logger.info("✅ Model training complete")
    prediction = predict_duration_task(X_train, model)
    train_me = evaluate_model_task(y_train, prediction)
    logger.info(f"✅ evaluate:{train_me}")   