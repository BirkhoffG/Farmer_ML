from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
import numpy as np
import pandas as pd


def load_file(filename):
    print(f"Loading {filename}...")
    df = pd.read_csv(filename, index_col=0, parse_dates=True, low_memory=False, na_values=['', 'NR'])
    # df.replace(to_replace='NR', value=np.NaN, inplace=True)
    return df


def load_files(path, crop, features=None):
    if features is None:
        features = ['price', 'volume', 'max_price', 'min_price']
    return (load_file(f"{path}/{feature}_{crop}.csv") for feature in features)


def drop_list(df: pd.DataFrame) -> []:
    """
    return the list that should be discarded due to the low subject frequency
    :param df: passing dataframe
    :return: columns to be dropped
    """
    return [df.columns[i] for i, f in enumerate(df.isna().sum() / len(df.index)) if f > 0.9]


def df2np(df)->np.array:
    print(f"converting to np array...")
    m = df.to_numpy(dtype=np.float64).transpose()
    print(f"matrix shape: {m.shape}")

    return m


def collab_SVD_Imputation(m: np.array)->np.array:
    biscaler = BiScaler()
    softimpute = SoftImpute()

    m_incomplete_normalized = biscaler.fit_transform(m)
    m_filled_softimpute = softimpute.fit_transform(m_incomplete_normalized)
    m_imputed = biscaler.inverse_transform(m_filled_softimpute)

    return m_imputed


def np2df(m, col):
    return pd.DataFrame(index=pd.date_range('2008-1-1', '2018-12-31'),
                        data=m.transpose(), columns=col)


if __name__ == '__main__':
    crop = 'Brinjal'
    path = '../dataset'
    # load datasets
    # price_df, volume_df = load_file(path=f"../dataset/", crop=crop)
    price_df, volume_df, max_df, min_df = load_file(f"{path}/price_{crop}.csv"), \
                                          load_file(f"{path}/volume_{crop}.csv"), \
                                          load_file(f"{path}/max_price_{crop}.csv"), \
                                          load_file(f"{path}/min_price_{crop}.csv")
    # drop list
    drop_list = drop_list(price_df)
    price_df, volume_df, max_df, min_df = price_df.drop(columns=drop_list), volume_df.drop(columns=drop_list),\
                                          max_df.drop(columns=drop_list), min_df.drop(columns=drop_list)
    # convert df to np.array
    price_m, volume_m, max_m, min_m = df2np(price_df), df2np(volume_df), df2np(max_df), df2np(min_df)
    # impute matrix
    price_imputed = collab_SVD_Imputation(price_m)
    volume_imputed = collab_SVD_Imputation(volume_m)
    max_imputed = collab_SVD_Imputation(max_m)
    min_imputed = collab_SVD_Imputation(min_m)
    # save imputed dataset
    np2df(m=price_imputed, col=price_df.columns).to_csv(f'{path}/price_imputed_{crop}.csv')
    np2df(m=volume_imputed, col=volume_df.columns).to_csv(f'{path}/volume_imputed_{crop}.csv')
    np2df(m=max_imputed, col=max_df.columns).to_csv(f'{path}/max_imputed_{crop}.csv')
    np2df(m=min_imputed, col=min_df.columns).to_csv(f'{path}/min_imputed_{crop}.csv')

    # print("loading price_df")
    # price_df = load_file(f'{path}/{features[0]}_{crop}.csv')
    # price_df = pd.read_csv(f'{path}price_{crop}.csv', index_col=0, parse_dates=True, low_memory=False)
    # print("loading volume_df")
    # volume_df = pd.read_csv(f'{path}volume_{crop}.csv', index_col=0, parse_dates=True, low_memory=False)
    # print("replacing NR -> NaN")
    # price_df.replace(to_replace='NR', value=np.NaN, inplace=True)
    # volume_df.replace(to_replace='NR', value=np.NaN, inplace=True)
    # print("Files loaded.")
    #
    # print(f"total length: {len(price_df.columns)}")
    # drop_list = []
    # days_n = len(price_df.index)
    #
    # for i, f in enumerate(price_df.isna().sum() / days_n):
    #     if f > 0.9:
    #         drop_list.append(price_df.columns[i])
    #
    # print(f"drop length: {len(drop_list)}")
    #
    # price_df_ = price_df.drop(columns=drop_list)
    # volume_df_ = volume_df.drop(columns=drop_list)
    #
    # print(f"remaining length: {len(price_df_.columns)}")
    # return price_df_, volume_df_
    #
    #
    #
    #
    #
    #
