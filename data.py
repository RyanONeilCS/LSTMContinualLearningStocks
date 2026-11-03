"""
Data loading and preprocessing

CSVs are hosted on GitHub so the dataset does not ship with the submission
First run needs internet and every run after uses the cache

Order of steps
    1 build the features and target from the raw prices
    2 split chronologically with the oldest days for training
    3 fit the scaler on training days only then apply to both
    4 build the windows inside each split on its own

2 before 3 stops the test set leaking into training
4 stops a window holding days from both splits
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests

#Public location of the dataset and no account is needed to reach it
BASE_URL = "https://raw.githubusercontent.com/RyanONeilCS/MachineLearningProjectData/main"

#Local folder the files are cached in
CACHE_DIR = "data_cache"

#Twelve companies in sector pairs so the forgetting matrix is readable
TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "GS", "XOM",
           "CVX", "JNJ", "PFE", "WMT", "KO", "BA"]

#Which column build_features shifts into the target
#daily_return is tomorrows price move and daily_range is tomorrows volatility
#main reads this into CONFIG so the log can never disagree with the code
TARGET = "daily_range"


#Download one ticker's CSV or read it from the cache
#Hosted on GitHub since it cannot ship with the submission
#Cache means it only downloads once
def get_Data(ticker: str):

    #Make sure the cache folder exists
    os.makedirs(CACHE_DIR, exist_ok = True)
    path = os.path.join(CACHE_DIR, f"{ticker}_data.csv")

    #Only hit the network if this ticker is not cached
    if not os.path.exists(path):
        response = requests.get(f"{BASE_URL}/{ticker}_data.csv", timeout = 30)
        response.raise_for_status()
        with open(path, "w", encoding = "utf-8", newline = "") as file:
            file.write(response.text)

    return pd.read_csv(path)


#Turn one ticker's raw OHLCV into the 3 features and the target
#Price level is never a feature and only changes are
#Target is shifted -1 so it is tomorrow and that shift is the task
#Drops the first and last row since one has no yesterday and one no tomorrow
def build_features(df: pd.DataFrame, ticker: str):

    #Sort by date so the sequence is in the right order
    df = df.sort_values("date").reset_index(drop = True)

    #Check the data is clean first
    assert df["date"].is_unique, f"{ticker} has duplicate dates"
    assert (df["close"] > 0).all(), f"{ticker} has a close price of zero or less"
    assert (df["volume"] > 0).all(), f"{ticker} has a volume of zero"

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    #Feature 1 how much the price moved today
    daily_return = close.pct_change()

    #Feature 2 how far it travelled compared to the close
    daily_range = (high - low) / close

    #Feature 3 how much the trading interest changed
    volume_change = np.log(volume).diff()

    #Target is tomorrow and this shift is the prediction task
    #Switching target means changing TARGET at the top and nothing else
    assert TARGET in ("daily_return", "daily_range"), f"unknown target {TARGET}"

    if TARGET == "daily_range":
        target = daily_range.shift(-1)
    else:
        target = daily_return.shift(-1)

    features = pd.DataFrame({
        "daily_return": daily_return,
        "daily_range": daily_range,
        "volume_change": volume_change,
    })

    #Drop the first and last row
    #Slicing keeps the remaining days contiguous
    features = features.iloc[1:-1]
    target = target.iloc[1:-1]

    features = features.to_numpy(dtype = float)
    target = target.to_numpy(dtype = float)

    #An unadjusted stock split shows up as a huge fake return
    assert np.isfinite(features).all(), f"{ticker} has a non finite feature"
    assert np.isfinite(target).all(), f"{ticker} has a non finite target"
    assert np.abs(target).max() < 0.30, f"{ticker} has a daily move over 30 percent"

    return features, target


#Cut days into overlapping windows and each one is its own sample
#Reads window days and predicts the day after the last one
#From N days you get N - window + 1 windows
#y must be reshaped to (N, 1) or the loss silently broadcasts to (B, B)
def make_windows(features: np.ndarray, target: np.ndarray, window: int, stride: int):

    windows = []
    targets = []

    #Each window reads window days and predicts the one after
    for i in range(0, len(features) - window + 1, stride):
        windows.append(features[i:i + window])
        targets.append(target[i + window - 1])

    X = np.asarray(windows, dtype = float)

    #Reshape is not optional or a flat y broadcasts to (B, B)
    y = np.asarray(targets, dtype = float).reshape(-1, 1)

    return X, y


#Load every ticker and stack them into one training and test set
#Used by the pooled model which cannot tell the companies apart
def load(tickers: list, window: int = 60, stride: int = 1, split: float = 0.8, verbose: bool = True):

    per_ticker = load_each(tickers, window, stride, split, verbose)

    #Stack the stocks and every window is its own sample
    X_train = np.concatenate([per_ticker[t][0] for t in tickers], axis = 0)
    y_train = np.concatenate([per_ticker[t][1] for t in tickers], axis = 0)
    X_test = np.concatenate([per_ticker[t][2] for t in tickers], axis = 0)
    y_test = np.concatenate([per_ticker[t][3] for t in tickers], axis = 0)

    return X_train, y_train, X_test, y_test


#Load every ticker but keep each company separate
#Needed for the incremental experiment and the forgetting matrix
#Order matters so split first then scale or the test set leaks in
#Scaler is fit on training days only and one per stock
#Windows are built inside each split so none straddle the boundary
def load_each(tickers: list, window: int = 60, stride: int = 1, split: float = 0.8, verbose: bool = True):

    per_ticker = {}

    for ticker in tickers:

        features, target = build_features(get_Data(ticker), ticker)

        #Split chronologically with the oldest days for training
        cut = int(len(features) * split)
        features_train, features_test = features[:cut], features[cut:]
        target_train, target_test = target[:cut], target[cut:]

        #Standardise using the training days only so the test set never leaks in
        #Each stock needs its own scaler since their sizes differ
        mean = features_train.mean(axis = 0)
        std = features_train.std(axis = 0)
        std[std == 0] = 1.0
        features_train = (features_train - mean) / std
        features_test = (features_test - mean) / std

        #Window each split on its own so no window straddles the boundary
        X_train, y_train = make_windows(features_train, target_train, window, stride)
        X_test, y_test = make_windows(features_test, target_test, window, stride)

        if verbose:
            print(f"{ticker:5s} train {X_train.shape}  test {X_test.shape}")

        per_ticker[ticker] = (X_train, y_train, X_test, y_test)

    return per_ticker
