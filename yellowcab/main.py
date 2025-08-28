import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import loguru
from yellowcab import data
from yellowcab.data import load_data

from yellowcab import task
from yellowcab.task import train



logger = loguru.logger



# def init():
#   # DATAS PROCESSING  ---------------
#   logger.info("... Data loaging")  
#   train_df = load_data(data.train_path)
#   #print(train_df.head())
#   logger.info("✅ Data loading complete")  

#   logger.info("... Data computing")  
#   train_df = compute_target(train_df)
#   sns.histplot(train_df["duration"], bins=100)
#   plt.show()
#   #print(train_df["duration"].describe())
#   logger.info("✅ Data Compute complete")  

#   train_df = filter_outliers(train_df)
#   sns.histplot(train_df["duration"], bins=100)
#   plt.title("Seaborn Demo: Tips Dataset")
#   plt.show()
#   logger.info("✅ Data filter complete")  

#   # DATA PREPARE
#   train_df = encode_categorical_cols(train_df)
#   logger.info("✅ Data Prepare complete")  

#   # DATA EXTRACT
#   logger.info("Data extract") 
#   X_train, y_train, dv = extract_x_y(train_df)
#   logger.info("✅ Data extract complete")  
#   # ----------------------------------------

#   # MODEL --------------------------------
#   logger.info("Data Model linear creation") 
#   linearmodel = create_linearreg_model()
#   logger.info("Data train") 
#   model = train_model(linearmodel, X_train, y_train)
#   logger.info("Data predict")  
#   prediction = predict_duration(X_train, model)
#   logger.info("Data eval")  
#   train_me = evaluate_model(y_train, prediction)
#   logger.info("✅ Model training complete")  
#   # ----------------------------------------

 
if __name__ == "__main__" :
#  load_datas() plus besoin deja la
  train()

  
 
