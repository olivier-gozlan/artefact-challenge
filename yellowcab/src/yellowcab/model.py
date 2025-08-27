import pandas as pd
import numpy as np
import pickle
from yellowcab.registry import load_model

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

N_ESTIMATORS = 50
MODEL_FOLDER = "./models"
forest_path = f"{MODEL_FOLDER}/forest_model.pkl"

def create_linearreg_model():
  return LinearRegression()

def instantiate_forest_model(fit=True):
  if fit:
    forest = RandomForestRegressor(n_estimators=N_ESTIMATORS
      , random_state=42
      ,verbose=2
      ,max_depth=10)
    return forest
  else:
    return load_model("forest_model")

      




