from DataPreprocessing.PrepareTrainTestArray import divide_train_test
from DataPreprocessing import CROPS
import numpy as np
import os
import pandas as pd

import random
# TODO generate market train/test array


def load_df(path, crop, features=('price', 'volume')):
    return pd.read_csv(f'{path}/{features[0]}_imputed_{crop}.csv', index_col=0, parse_dates=True, low_memory=False), \
           pd.read_csv(f'{path}/{features[1]}_imputed_{crop}.csv', index_col=0, parse_dates=True, low_memory=False)


def resample(data: pd.DataFrame, scale):
    return data.resample(scale).mean()


def create_multi_ahead_samples(ts: np.array, look_back: int, look_ahead=1):

    '''
    :param ts: input ts np array
    :param look_back: history window size
    :param look_ahead: forecasting window size
    :return: trainx with shape (sample_num, look_back, 1) or and trainy with shape (sample_num, look_ahead)
    '''

    x, y = [], []
    for i in range(len(ts) - look_back - look_ahead):
        start, mid, end = i, i + look_back, i + look_back + look_ahead
        history_seq = ts[start: mid]
        future_seq = ts[mid: end]
        x.append(history_seq)
        y.append(future_seq)
    data_x = np.array(x)
    data_y = np.array(y)
    return data_x, data_y


def prepare_x_y(ts, lag, h_train):
    x, y = create_multi_ahead_samples(ts, lag, h_train)
    y = np.squeeze(y).reshape(-1, h_train)
    return x, y


def one_hot_encoding(arr: np.array):
    encoded_arr = np.zeros((len(arr), arr.max()+1))
    print(f"encoded_arr size: {encoded_arr.shape}")
    encoded_arr[np.arange(len(arr)), arr] = 1
    return encoded_arr


def prepare_array(df, lag, ahead):
    # date index
    time_seq = df.index.dayofyear - 1
    # convert to one-hot encoding matrix
    ts = df[df.columns[0]].to_numpy()

    ts = np.expand_dims(ts, axis=1)
    print(f"ts shape: {ts.shape}; time_seq shape: {time_seq.shape}")
    ts_seq = np.concatenate((ts, time_seq), axis=-1)
    train_x, train_y, test_x, test_y = divide_train_test(ts_seq, lag, ahead, ahead)

    for ix, col in enumerate(df.columns):
        print(f"[{ix+1}/{len(df.columns)}] col: {col}")
        if ix == 0: continue

        ts = df[df.columns[0]].to_numpy()
        ts_seq = np.concatenate((ts, time_seq), axis=-1).astype(int)
        temp_train_x, temp_train_y, temp_test_x, temp_test_y = \
            divide_train_test(ts_seq, lag=lag, h_train=ahead, h_test=ahead)

        train_x = np.concatenate((train_x, temp_train_x), axis=0)
        train_y = np.concatenate((train_y, temp_train_y), axis=0)
        test_x = np.concatenate((test_x, temp_test_x), axis=0)
        test_y = np.concatenate((test_y, temp_test_y), axis=0)
        # print("=" * 6)
        print(f"train_x size: {train_x.shape}")
        print(f"train_y size: {train_y.shape}")
        print(f"test_x size: {test_x.shape}")
        print(f"test_y size: {test_y.shape}")
    return train_x, train_y, test_x, test_y


def save_arrays(path, feature, **kwargs):
    try:
        for file_name, arr in kwargs.items():
            print(f"saving {path}/{feature[:3]}_{file_name}.npy")
            np.save(f"{path}/{feature[:3]}_train_x.npy", arr)

    except FileNotFoundError:
        os.mkdir(path)
        print(f"Create path: {path}")
        save_arrays(path, feature, **kwargs)

#%%


def process(crop, lag=90, ahead=1, rescale='4d'):
    # parameters
    # ahead = 1
    # lag = 90
    # crop = CROPS[-1]

    # load df
    price_df, volume_df = load_df(path="../dataset/imputed", crop=crop, features=('price', 'volume'))

    # resample
    price_df, volume_df = resample(price_df, rescale), resample(volume_df, rescale)

    print(f"Crop: {crop}")
    print("Generating price arr...")
    price_train_x, price_train_y, price_test_x, price_test_y = prepare_array(price_df, lag, ahead, apply_one_hot_encoding=False)
    assert len(price_train_x) == len(price_train_y); assert len(price_test_x) == len(price_test_y)

    pretrain_len = len(price_train_x) // 4 * 3
    price_pretrain_x, price_pretrain_y = price_train_x[:pretrain_len], price_train_y[:pretrain_len]

    print("Generating volume arr... ")
    volume_train_x, volume_train_y, volume_test_x, volume_test_y = prepare_array(volume_df, lag, ahead)
    assert len(volume_train_x) == len(volume_train_y); assert len(volume_test_x) == len(volume_test_y)

    pretrain_len = len(volume_train_x) // 4 * 3
    volume_pretrain_x, volume_pretrain_y = volume_train_x[:pretrain_len], volume_train_y[:pretrain_len]

    path = f'../np_array/pretrain/{crop}/train_90_01_test_90_01/pretrain'
    save_arrays(path=path, feature='price', train_x=price_pretrain_x, train_y=price_pretrain_y)
    save_arrays(path=path, feature='volume', train_x=volume_pretrain_x, train_y=volume_pretrain_y)

    path = f'../np_array/pretrain/{crop}/train_90_01_test_90_01/model'
    save_arrays(path=path, feature='price', train_x=price_train_x, train_y=price_train_y,
                test_x=price_test_x, test_y=price_test_y)
    save_arrays(path=path, feature='volume', train_x=volume_train_x, train_y=volume_train_y,
                test_x=volume_test_x, test_y=volume_test_y)


if __name__ == '__main__':
    for crop in [CROPS[0], CROPS[1], CROPS[-1]]:
        process(crop)
