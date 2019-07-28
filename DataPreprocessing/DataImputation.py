from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
import numpy as np
import pandas as pd


def load_file(path, crop):
    print("loading price_df")
    price_df = pd.read_csv(f'{path}price_{crop}.csv', index_col=0, parse_dates=True, low_memory=False)
    print("loading volume_df")
    volume_df = pd.read_csv(f'{path}volume_{crop}.csv', index_col=0, parse_dates=True, low_memory=False)
    print("replacing NR -> NaN")
    price_df.replace(to_replace='NR', value=np.NaN, inplace=True)
    volume_df.replace(to_replace='NR', value=np.NaN, inplace=True)
    print("Files loaded.")

    print(f"total length: {len(price_df.columns)}")
    drop_list = []
    days_n = len(price_df.index)

    for i, f in enumerate(price_df.isna().sum() / days_n):
        if f > 0.9:
            drop_list.append(price_df.columns[i])

    print(f"drop length: {len(drop_list)}")

    price_df_ = price_df.drop(columns=drop_list)
    volume_df_ = volume_df.drop(columns=drop_list)

    print(f"remaining length: {len(price_df_.columns)}")
    return price_df_, volume_df_


def df2np(df):
    print(f"converting to np array...")
    m = df.to_numpy(dtype=np.float64).transpose()
    print(f"matrix shape: {m.shape}")

    return m


def collab_SVD_Imputation(m: np.array()):
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
    # load datasets
    price_df, volume_df = load_file(path=f"../dataset/", crop=crop)
    # convert df to np.array
    price_m, volume_m = df2np(price_df), df2np(volume_df)
    # impute matrix
    price_imputed = collab_SVD_Imputation(price_m)
    volume_imputed = collab_SVD_Imputation(volume_m)
    # save imputed dataset
    np2df(m=price_imputed, col=price_df.columns).to_csv(f'../dataset/price_imputed_{crop}.csv')
    np2df(m=volume_imputed, col=volume_df.columns).to_csv(f'../dataset/volume_imputed_{crop}.csv')

