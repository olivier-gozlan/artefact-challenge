import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

N_ESTIMATORS = 50
MODEL_FOLDER = "./models"
forest_path = f"{MODEL_FOLDER}/forest_model.pkl"

def create_linearreg_model():
  return LinearRegression()

def create_forest_model():
  forest = RandomForestRegressor(n_estimators=N_ESTIMATORS
      , random_state=42
      ,verbose=2
      ,max_depth=10)
  return forest

def load_forest_model():
    with open(forest_path, "rb") as f:
        return pickle.load(f)




