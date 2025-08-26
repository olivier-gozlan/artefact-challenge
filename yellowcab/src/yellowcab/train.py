import pandas as pd
import numpy as np

from sklearn.metrics import root_mean_squared_error

from scipy.sparse import csr_matrix

def train_model(model, x_train: csr_matrix, y_train: np.ndarray):
    model.fit(x_train, y_train)
    return model


def predict_duration(input_data: csr_matrix, model):
    return model.predict(input_data)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray):
    return root_mean_squared_error(y_true, y_pred)

