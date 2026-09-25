# src/data/loader.py
import numpy as np
import pandas as pd

from src.config import RAW_DIR


def merge_market_data(sp500, vix, nifty):
    """Merge daily closes onto the S&P 500 calendar without inventing data.

    Returns are computed on each market's own calendar. After Nifty coverage begins, an S&P date
    with no Nifty return means the Indian market was closed: Nifty_Ret = 0.0, Nifty_Closed = 1.
    Before coverage begins (Yahoo's ^NSEI history starts in Sep 2007) nothing is known, so both
    columns stay NaN instead of being zero-filled; XGBoost handles NaN natively.
    """
    sp = sp500[["Close"]].rename(columns={"Close": "Price"})
    vx = vix[["Close"]].rename(columns={"Close": "VIX"})
    nf = nifty[["Close"]].rename(columns={"Close": "NIFTY"})

    sp["Log_Ret"] = np.log(sp["Price"] / sp["Price"].shift(1))
    nf["Nifty_Ret"] = np.log(nf["NIFTY"] / nf["NIFTY"].shift(1))

    df = sp.join(vx, how="left").join(nf[["Nifty_Ret"]], how="left")
    df["VIX"] = df["VIX"].ffill()  # VIX is a level, so carrying it forward is safe

    coverage_start = nf["Nifty_Ret"].first_valid_index()
    covered = df.index >= coverage_start if coverage_start is not None else np.zeros(len(df), bool)
    df["Nifty_Closed"] = np.where(covered, df["Nifty_Ret"].isna().astype(float), np.nan)
    df.loc[covered, "Nifty_Ret"] = df.loc[covered, "Nifty_Ret"].fillna(0.0)

    return df.dropna(subset=["Log_Ret", "VIX"])


def _download_close(ticker, start):
    import yfinance as yf  # imported lazily so merge_market_data is testable offline

    data = yf.download(ticker, start=start, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def fetch_data(start_date="2000-01-01"):
    """Downloads S&P 500, VIX and Nifty 50 and saves the merged raw dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Global Market Data to: {RAW_DIR}")
    frames = {name: _download_close(t, start_date) for name, t in
              [("sp500", "^GSPC"), ("vix", "^VIX"), ("nifty", "^NSEI")]}
    master_df = merge_market_data(frames["sp500"], frames["vix"], frames["nifty"])

    save_path = RAW_DIR / "global_markets.csv"
    master_df.to_csv(save_path)
    first_nifty = master_df["Nifty_Ret"].first_valid_index()
    print(f"   Saved Global Data. Final shape: {master_df.shape}")
    print(f"   Dataset runs {master_df.index.min().date()} to {master_df.index.max().date()}; "
          f"Nifty data starts {first_nifty.date() if first_nifty is not None else 'never'}")


if __name__ == "__main__":
    fetch_data()