import pandas as pd
import numpy as np
import os

def split_data():
    '''we will load features.csv and then remove missing data rows and then finally split into train and test split'''
    print("splitting starting.....")
    input_path = os.path.join("data",'processed', "features.csv")
    if not os.path.exists(input_path):
        print("features file not found check location or file path!!!!")
    
    df = pd.read_csv(input_path,index_col=0,parse_dates=True)
    df = df.dropna()
    split_index = int(0.8*len(df))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    train_df.to_csv(os.path.join("data","processed","train.csv"))
    test_df.to_csv(os.path.join("data", "processed", "test.csv")) 
    print(f"Saved train.csv ({len(train_df)} rows) and test.csv ({len(test_df)} rows)")

if __name__ == "__main__":
    split_data()