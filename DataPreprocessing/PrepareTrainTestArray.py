from src.util import load_data, createSamples, divideTrainTest, create_multi_ahead_samples
from src.ts_loader import Time_Series_Data
from src.NN_forecasting import single_model_forecasting
from src.NN_train import train, predict, predict_iteration
from src import eval

import pandas as pd
import numpy as np
from pathlib import Path


def load_df(path, crop):
    return pd.read_csv(f'{path}price_imputed_{crop}.csv', index_col=0, parse_dates=True, low_memory=False), \
           pd.read_csv(f'{path}volume_imputed_{crop}.csv', index_col=0, parse_dates=True, low_memory=False)


def resample(data: pd.DataFrame, scale):
    return data.resample(scale).mean()


def divide_train_test(data, lag, h_train, h_test):
    train_data, test_data = divideTrainTest(data)

    train_x, train_y = create_multi_ahead_samples(train_data, lag, h_train, RNN=True)
    test_x, test_y = create_multi_ahead_samples(test_data, lag, h_test, RNN=True)

    train_y = np.squeeze(train_y).reshape(-1, h_train)
    test_y = np.squeeze(test_x).reshape(-1, h_test)
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    price_df, volume_df = load_df(path="../dataset/", crop='Brinjal')

    price_df = resample(data=price_df, scale='4d')

    h_train, h_test = 30, 30
    lag = 90

    price_train_x, price_train_y, price_test_x, price_test_y = \
        divide_train_test(price_df, lag, h_train, h_test)

