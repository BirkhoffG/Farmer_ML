from src.util import load_data, createSamples, divideTrainTest, create_multi_ahead_samples
from src.ts_loader import Time_Series_Data
from src.NN_forecasting import single_model_forecasting
from src.NN_train import train, predict, predict_iteration
from src import eval

import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_df(path, crop, features=('price', 'volume')):
    return pd.read_csv(f'{path}/{features[0]}_imputed_{crop}.csv', index_col=0, parse_dates=True, low_memory=False), \
           pd.read_csv(f'{path}/{features[1]}_imputed_{crop}.csv', index_col=0, parse_dates=True, low_memory=False)


def resample(data: pd.DataFrame, scale):
    return data.resample(scale).mean()


def divide_train_test(data, lag, h_train, h_test):
    train_data, test_data = divideTrainTest(data)

    train_x, train_y = create_multi_ahead_samples(train_data, lag, h_train, RNN=True)
    test_x, test_y = create_multi_ahead_samples(test_data, lag, h_test, RNN=True)

    train_y = np.squeeze(train_y).reshape(-1, h_train)
    test_y = np.squeeze(test_y).reshape(-1, h_test)
    return train_x, train_y, test_x, test_y


def concatenate_array(df, lag, h_train, h_test):
    train_x, train_y, test_x, test_y = \
        divide_train_test(df[df.columns[0]], lag, h_train, h_test)

    for ix, col in enumerate(df.columns):
        print(f"[{ix+1}/{len(df.columns)}] col: {col}")
        if ix == 0:
            continue
        # data = df[col].values.reshape(-1, 1).astype("float32")
        temp_train_x, temp_train_y, temp_test_x, temp_test_y = divide_train_test(df[col], lag=lag, h_train=h_train, h_test=h_test)
        train_x = np.concatenate((train_x, temp_train_x), axis=0)
        train_y = np.concatenate((train_y, temp_train_y), axis=0)
        test_x = np.concatenate((test_x, temp_test_x), axis=0)
        test_y = np.concatenate((test_y, temp_test_y), axis=0)
    return train_x, train_y, test_x, test_y


def save_arrays(path, feature, train_x, train_y, test_x, test_y):
    try:
        print(f"saving {path}/{feature[:3]}_train_x.npy")
        np.save(f"{path}/{feature[:3]}_train_x.npy", train_x)

        print(f"saving {path}/{feature[:3]}_train_y.npy")
        np.save(f"{path}/{feature[:3]}_train_y.npy", train_y)

        print(f"saving {path}/{feature[:3]}_test_x.npy")
        np.save(f"{path}/{feature[:3]}_test_x.npy", test_x)

        print(f"saving {path}/{feature[:3]}_test_y.npy")
        np.save(f"{path}/{feature[:3]}_test_y.npy", test_y)
    except FileNotFoundError:
        os.mkdir(path)
        print(f"Create path: {path}")
        save_arrays(path, feature, train_x, train_y, test_x, test_y)


def correct_max_min(arr: np.array, max_arr: np.array, min_arr: np.array):
    return np.where(max_arr > arr, max_arr, arr), np.where(min_arr < arr, min_arr, arr)


#%%
if __name__ == '__main__':
    price_df, volume_df = load_df(path="../dataset", crop="Brinjal", features=('price', 'volume'))
    min_price_df, max_price_df = load_df(path="../dataset", crop='Brinjal', features=('min', 'max'))

    print("Resample price_df...")
    price_df = resample(data=price_df, scale='4d')
    print("Resample volume_df...")
    volume_df = resample(data=volume_df, scale='4d')
    print("Resample max_df...")
    max_price_df = resample(data=max_price_df, scale='4d')
    print("Resample min_df...")
    min_price_df = resample(data=min_price_df, scale='4d')

    h_train, h_test = 1, 1
    lag = 90

    print("Generating price train/test arr...")
    price_train_x, price_train_y, price_test_x, price_test_y = \
        concatenate_array(price_df, lag, h_train, h_test)

    print("Generating volume train/test arr... ")
    volume_train_x, volume_train_y, volume_test_x, volume_test_y = \
        concatenate_array(volume_df, lag, h_train, h_test)

    print("Generating max price train/test arr... ")
    max_train_x, max_train_y, max_test_x, max_test_y = \
        concatenate_array(max_price_df, lag, h_train, h_test)

    print("Generating min price train/test arr... ")
    min_train_x, min_train_y, min_test_x, min_test_y = \
        concatenate_array(min_price_df, lag, h_train, h_test)

    print("Correcting train_x arr...")
    max_train_x, min_train_x = correct_max_min(price_train_x, max_train_x, min_train_x)
    assert (max_train_x >= price_train_x).all()
    assert (min_train_x <= price_train_x).all()

    print("Correcting train_y arr...")
    max_train_y, min_train_y = correct_max_min(price_train_y, max_train_y, min_train_y)
    assert (max_train_y >= price_train_y).all()
    assert (min_train_y <= price_train_y).all()

    print("Correcting test_x arr...")
    max_test_x, min_test_x = correct_max_min(price_test_x, max_test_x, min_test_x)
    assert (max_test_x >= price_test_x).all()
    assert (min_test_x <= price_test_x).all()

    print("Correcting test_y arr...")
    max_test_y, min_test_y = correct_max_min(price_test_y, max_test_y, min_test_y)
    assert (max_test_y >= price_test_y).all()
    assert (min_test_y <= price_test_y).all()

    path = '../np_array/train_90_01_test_90_01'
    save_arrays(path=path, feature='price', train_x=price_train_x, train_y=price_train_y,
                test_x=price_test_x, test_y=price_test_y)
    save_arrays(path=path, feature='volume', train_x=volume_train_x, train_y=volume_train_y,
                test_x=volume_test_x, test_y=volume_test_y)
    save_arrays(path=path, feature='max_price', train_x=max_train_x, train_y=max_train_y,
                test_x=max_test_x, test_y=max_test_y)
    save_arrays(path=path, feature='min_price', train_x=min_train_x, train_y=min_train_y,
                test_x=min_test_x, test_y=min_test_y)


