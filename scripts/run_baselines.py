import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.naive import NaiveModel
from src.models.linear_lag import LinearBaseline
from src.models.garch import GarchModel

def run_fight():
    print("strating with the comparision")
    # Load Data
    try:
        train_df = pd.read_csv("data/processed/train.csv", index_col=0, parse_dates=True)
        test_df = pd.read_csv("data/processed/test.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Data not found. Run src/data/splitter.py")
        return

    print(f"   Train: {len(train_df)} | Test: {len(test_df)}")
    print("-" * 30)
    # Naive
    print("\nNaive Model")
    naive = NaiveModel()
    print(f"   Naive RMSE:  {naive.evaluate(test_df)['RMSE']:.5f}")
    #Linear
    print("\nLinear Model")
    linear = LinearBaseline()
    linear.train(train_df)
    print(f"   Linear RMSE: {linear.evaluate(test_df)['RMSE']:.5f}")
    #GARCH
    print("\nGARCH Model")
    if 'Log_Ret' in train_df.columns:
        garch = GarchModel()
        print(f"GARCH RMSE:  {garch.evaluate(train_df, test_df)['RMSE']:.5f}")
    else:
        print(" Skipped: 'Log_Ret' missing.")

if __name__ == "__main__":
    run_fight()