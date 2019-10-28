# from DataPreprocessing.PrepareTrainTestArray import divide_train_test
#from __init__ import CROPS
from DataPreprocessing import CROPS
# from src.util import divideTrainTest
import numpy as np
import json
import os
import pandas as pd

import random


def load_df(path, crop, features=('price', 'volume')):
    return pd.read_csv(f'{path}/{features[0]}_imputed_{crop}.csv', index_col=0, parse_dates=True, low_memory=False), \
           pd.read_csv(f'{path}/{features[1]}_imputed_{crop}.csv', index_col=0, parse_dates=True, low_memory=False)


def resample(data: pd.DataFrame, scale):
    return data.resample(scale).mean()


def create_multi_ahead_samples(ts: np.array, look_back: int, look_ahead=1, mktid=None, lon=None, lat=None):
    '''
    :param ts: input ts np array
    :param look_back: history window size
    :param look_ahead: forecasting window size
    :return: trainx with shape (sample_num, look_back, 1) or and trainy with shape (sample_num, look_ahead)
    '''

    x, y, mkt, geo = [], [], [], []
    for i in range(len(ts) - look_back - look_ahead):
        start, mid, end = i, i + look_back, i + look_back + look_ahead
        history_seq = ts[start: mid]
        future_seq = ts[mid: end]
        x.append(history_seq)
        y.append(future_seq)
        mkt.append(mktid)
        geo.append([lon, lat])
    data_x = np.array(x)
    data_y = np.array(y)
    data_mkt = np.array(mkt)
    data_geo = np.array(geo)
    return data_x, data_y, data_mkt, data_geo


def one_hot_encoding(arr: np.array):
    arr = np.array(arr, dtype = int)
    print('arr.max()', arr.max())
    print('len(arr)', len(arr))
    encoded_arr = np.zeros((len(arr), arr.max() + 1))
    print(f"encoded_arr size: {encoded_arr.shape}")
    encoded_arr[np.arange(len(arr)), arr] = 1
    return encoded_arr


def divide_train_test(data, lag, h_train, h_test, mktid, lon, lat):
    # train size: 75% train set; 25% test set
    train_size = int(len(data) * 0.75)

    train_data, test_data = data[0:train_size], data[train_size:]

    train_x, train_y, train_mkt, train_geo = create_multi_ahead_samples(train_data, lag, h_train, mktid=mktid,
                                                                        lon=lon, lat=lat)
    test_x, test_y, test_mkt, test_geo = create_multi_ahead_samples(test_data, lag, h_test, mktid=mktid,
                                                                    lon=lon, lat=lat)

    train_y = np.squeeze(train_y).reshape(-1, h_train)
    test_y = np.squeeze(test_y).reshape(-1, h_test)
    return train_x, train_y, test_x, test_y, train_mkt, test_mkt, train_geo, test_geo


def prepare_array(df, lag, ahead, mkt2loc: dir()):
    # convert to one-hot encoding matrix
    ts = df[df.columns[0]].to_numpy()

    ts = np.expand_dims(ts, axis=1)
    train_x, train_y, test_x, test_y, train_mkt, test_mkt, train_geo, test_geo = \
        divide_train_test(ts, lag, ahead, ahead,
                          mkt2loc[df.columns[0]]['id'],
                          lat=mkt2loc[df.columns[0]]['lat'],
                          lon=mkt2loc[df.columns[0]]['lng'])

    for ix, col in enumerate(df.columns):
        #print(f"[{ix + 1}/{len(df.columns)}] col: {col}")
        if ix == 0: continue

        ts = np.expand_dims(df[col].to_numpy(), axis=1)
        # ts_seq = np.concatenate((ts, time_seq), axis=-1).astype(int)
        temp_train_x, temp_train_y, temp_test_x, temp_test_y, temp_train_mkt, temp_test_mkt, temp_train_geo, temp_test_geo = \
            divide_train_test(ts, lag=lag, h_train=ahead, h_test=ahead,
                              mktid=mkt2loc[col]['id'],
                              lat=mkt2loc[col]['lat'],
                              lon=mkt2loc[col]['lng'])

        train_x = np.concatenate((train_x, temp_train_x), axis=0)
        train_y = np.concatenate((train_y, temp_train_y), axis=0)
        test_x = np.concatenate((test_x, temp_test_x), axis=0)
        test_y = np.concatenate((test_y, temp_test_y), axis=0)
        train_mkt = np.concatenate((train_mkt, temp_train_mkt), axis=0)
        test_mkt = np.concatenate((test_mkt, temp_test_mkt), axis=0)
        train_geo = np.concatenate((train_geo, temp_train_geo), axis=0)
        test_geo = np.concatenate((test_geo, temp_test_geo), axis=0)
        # print("=" * 6)
        # print(f"train_x size: {train_x.shape}")
        # print(f"train_y size: {train_y.shape}")
        # print(f"test_x size: {test_x.shape}")
        # print(f"test_y size: {test_y.shape}")
        # print(f"train_mkt size: {train_mkt.shape}")
        # print(f"test_mkt size: {test_mkt.shape}")
        # print(f"train_geo size: {train_geo.shape}")
        # print(f"test_geo size: {test_geo.shape}")

    return train_x, train_y, test_x, test_y, train_mkt, test_mkt, train_geo, test_geo


def save_arrays(path, feature, **kwargs):
    try:
        for file_name, arr in kwargs.items():
            print(f"saving {path}/{feature[:3]}_{file_name}.npy")
            np.save(f"{path}/{feature[:3]}_{file_name}.npy", arr)

    except FileNotFoundError:
        os.mkdir(path)
        print(f"Create path: {path}")
        save_arrays(path, feature, **kwargs)


def process(crop, lag=90, ahead=1, rescale='4d'):
    # load df
    price_df, volume_df = load_df(path="../dataset/imputed", crop=crop, features=('price', 'volume'))

    # mkt2loc
    with open('../mkt2loc.json', 'r') as f:
        mkt2loc = json.load(f)

    # resample
    price_df, volume_df = resample(price_df, rescale), resample(volume_df, rescale)

    print(f"Crop: {crop}")
    print("Generating price arr...")
    price_train_x, price_train_y, price_test_x, price_test_y, \
    price_train_mkt, price_test_mkt, price_train_geo, price_test_geo = \
        prepare_array(price_df, lag, ahead, mkt2loc)
    assert len(price_train_x) == len(price_train_y);
    assert len(price_test_x) == len(price_test_y)
    
    print('price_train_mkt.shape',price_train_mkt.shape)

    print("Generating volume arr... ")
    volume_train_x, volume_train_y, volume_test_x, volume_test_y, \
    volume_train_mkt, volume_test_mkt, volume_train_geo, volume_test_geo = \
        prepare_array(volume_df, lag, ahead, mkt2loc)
    assert len(volume_train_x) == len(volume_train_y);
    assert len(volume_test_x) == len(volume_test_y)

    path = f'../np_array/final/train_90_01_test_90_01/{crop}'
    save_arrays(path=path, feature='price', train_x=price_train_x, train_y=price_train_y,
                test_x=price_test_x, test_y=price_test_y, train_mkt=price_train_mkt, test_mkt=price_test_mkt,
                train_geo=price_train_geo, test_geo=price_test_geo)
    save_arrays(path=path, feature='volume', train_x=volume_train_x, train_y=volume_train_y,
                test_x=volume_test_x, test_y=volume_test_y, train_mkt=volume_train_mkt, test_mkt=volume_test_mkt,
                train_geo=volume_train_geo, test_geo=volume_test_geo)


if __name__ == '__main__':
    #for crop in [CROPS[0], CROPS[1], CROPS[-1]]:
    process(CROPS[0])
        
        