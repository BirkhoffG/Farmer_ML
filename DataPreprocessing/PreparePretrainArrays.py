from DataPreprocessing.PrepareTrainTestArray import divide_train_test, load_df, save_arrays, CROPS, resample
from src.util import create_multi_ahead_samples
import numpy as np
import os


def prepare_x_y(df, lag, h_train, RNN=True):
    x, y = create_multi_ahead_samples(df, lag, h_train, RNN=RNN)
    y = np.squeeze(y).reshape(-1, h_train)
    return x, y


def pretrained_array(df, lag, h_train, h_test, divide_year=2013):
    x, y = prepare_x_y(df.loc[:f"{divide_year}"][df.columns[0]], lag, h_train, RNN=True)
    train_x, train_y, test_x, test_y = divide_train_test(df.loc[f"{divide_year+1}":][df.columns[0]], lag, h_train, h_test)
    
    for ix, col in enumerate(df.columns):
        print(f"[{ix+1}/{len(df.columns)}] col: {col}")
        if ix == 0:
            continue
        temp_x, temp_y = prepare_x_y(df.loc[:f"{divide_year}"][col], lag=lag, h_train=h_train)
        x = np.concatenate((x, temp_x), axis=0)
        y = np.concatenate((y, temp_y), axis=0)

        temp_train_x, temp_train_y, temp_test_x, temp_test_y = \
            divide_train_test(df.loc[f"{divide_year+1}":][col], lag=lag, h_train=h_train, h_test=h_test)
        train_x = np.concatenate((train_x, temp_train_x), axis=0)
        train_y = np.concatenate((train_y, temp_train_y), axis=0)
        test_x = np.concatenate((test_x, temp_test_x), axis=0)
        test_y = np.concatenate((test_y, temp_test_y), axis=0)
        # print("=" * 6)
        # print(f"x size: {x.shape}")
        # print(f"y size: {y.shape}")
        # print(f"train_x size: {train_x.shape}")
        # print(f"train_y size: {train_y.shape}")
        # print(f"test_x size: {test_x.shape}")
        # print(f"test_y size: {test_y.shape}")
    return x, y, train_x, train_y, test_x, test_y


def save_pretrained_arrays(path, feature, train_x, train_y):
    try:
        print(f"saving {path}/{feature[:3]}_train_x.npy")
        np.save(f"{path}/{feature[:3]}_train_x.npy", train_x)

        print(f"saving {path}/{feature[:3]}_train_y.npy")
        np.save(f"{path}/{feature[:3]}_train_y.npy", train_y)

    except FileNotFoundError:
        os.mkdir(path)
        print(f"Create path: {path}")
        save_pretrained_arrays(path, feature, train_x, train_y)

#%%
if __name__ == '__main__':
    # parameters
    h_train, h_test = 1, 1
    lag = 90
    crop = CROPS[-1]

    # load df
    price_df, volume_df = load_df(path="../dataset/imputed", crop=crop, features=('price', 'volume'))

    # resample
    price_df, volume_df = resample(price_df, '4d'), resample(volume_df, '4d')

    print(f"Crop: {crop}")
    print("Generating price arr...")
    price_pretrain_x, price_pretrain_y, price_train_x, price_train_y, price_test_x, price_test_y = \
        pretrained_array(price_df, lag, h_train, h_test)

    print("Generating volume arr... ")
    volume_pretrain_x, volume_pretrain_y, volume_train_x, volume_train_y, volume_test_x, volume_test_y = \
        pretrained_array(volume_df, lag, h_train, h_test)

    path = f'../np_array/pretrain/{crop}/train_90_01_test_90_01/pretrain'
    save_pretrained_arrays(path=path, feature='price', train_x=price_pretrain_x, train_y=price_pretrain_y)
    save_pretrained_arrays(path=path, feature='volume', train_x=volume_pretrain_x, train_y=volume_pretrain_y)

    path = f'../np_array/pretrain/{crop}/train_90_01_test_90_01/model'
    save_arrays(path=path, feature='price', train_x=price_train_x, train_y=price_train_y,
                test_x=price_test_x, test_y=price_test_y)
    save_arrays(path=path, feature='volume', train_x=volume_train_x, train_y=volume_train_y,
                test_x=volume_test_x, test_y=volume_test_y)

