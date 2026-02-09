import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LinearBaseline:
    
    def __init__(self):
        self.model = LinearRegression()
        # We only use past volatility to predict future volatility
        self.features = ['Vol_5d_Lag_1', 'Vol_5d_Lag_2']

    def train(self, df_train):
        # Fit the line
        self.model.fit(df_train[self.features], df_train['Target_Vol_Next_5d'])

    def predict(self, df_test):
        return self.model.predict(df_test[self.features])

    def evaluate(self, df_test):
        y_true = df_test['Target_Vol_Next_5d']
        y_pred = self.predict(df_test)
        
        # Calculate Error
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        return {"RMSE": rmse, "MAE": mae}