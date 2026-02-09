import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error , mean_absolute_error

class NaiveModel:
    def __init__(self):
        pass
    def predict(self, df):
        return df['Vol_5d']
    def evaluate(self,df):
        y_true = df['Target_Vol_Next_5d']
        y_pred = self.predict(df)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        return {"RMSE": rmse, "MAE": mae}