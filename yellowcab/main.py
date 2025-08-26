import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from yellowcab import data
from yellowcab.data import load_datas, load_data, compute_target, filter_outliers, encode_categorical_cols, extract_x_y

def init():
  # DATAS PROCESSING  ---------------
  train_df = load_data(data.train_path)
  #print(train_df.head())

  train_df = compute_target(train_df)
  sns.histplot(train_df["duration"], bins=100)
  plt.show()
  #print(train_df["duration"].describe())

  train_df = filter_outliers(train_df)
  sns.histplot(train_df["duration"], bins=100)
  plt.title("Seaborn Demo: Tips Dataset")
  plt.show()

  # DATA PREPARE
  train_df = encode_categorical_cols(train_df)
  # DATA EXTRACT
  X_train, y_train, dv = extract_x_y(train_df)
  # ----------------------------------------

  # MODEL --------------------------------

  # ----------------------------------------

  print('so far so good')

if __name__ == "__main__" :
  load_datas()
  init()