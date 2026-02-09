import pandas as pd
import sys
import os

# This allows Python to find the 'src' folder from the 'scripts' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.naive import NaiveModel
from src.models.linear_lag import LinearBaseline

def run_fight():
    # 1. Load the features we built in Phase 2
    print("Loading features...")
    features_path = os.path.join("data", "processed", "features.csv")
    
    if not os.path.exists(features_path):
        print("Error: features.csv not found.")
        return
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    df = df.dropna()

    # 2. Split Data (Simple Time Split)
    # Train on the first 80% (History), Test on the last 20% (Future)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"   Train Rows: {len(train_df)}")
    print(f"   Test Rows:  {len(test_df)}")

    # --- ROUND 1: NAIVE MODEL ---
    print("\nRound 1: Naive Model")
    naive = NaiveModel()
    # Naive doesn't need training
    naive_scores = naive.evaluate(test_df)
    print(f"   Naive RMSE: {naive_scores['RMSE']:.5f} (The Score to Beat)")

    # --- ROUND 2: LINEAR MODEL ---
    print("\nRound 2: Linear Model")
    linear = LinearBaseline()
    linear.train(train_df)
    linear_scores = linear.evaluate(test_df)
    print(f"   Linear RMSE: {linear_scores['RMSE']:.5f}")

    # --- RESULT ---
    print("\nRESULTS")
    if linear_scores['RMSE'] < naive_scores['RMSE']:
        improvement = naive_scores['RMSE'] - linear_scores['RMSE']
        print(f"   Linear Wins! (Improved by {improvement:.5f})")
    else:
        print("   Naive Wins! (Simple is better).")

if __name__ == "__main__":
    run_fight()