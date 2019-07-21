import numpy as np
import pandas as pd
import impyute.imputation.cs.fast_knn as knn
from fancyimpute import BiScaler

crop = 'brinjal'
path = '/storage/home/a/auy212/work/'
# path = './'

# load the csv file
print("loading price_df")
price_df = pd.read_csv(f'{path}price_{crop}.csv', index_col=0, parse_dates=True, low_memory=False)
print("loading volume_df")
volume_df = pd.read_csv(f'{path}volume_{crop}.csv', index_col=0, parse_dates=True, low_memory=False)
price_df.replace(to_replace='NR', value=np.NaN, inplace=True)
volume_df.replace(to_replace='NR', value=np.NaN, inplace=True)

# Drop columns with observation's freqeuncy < 0.1

print(f"total length: {len(price_df.columns)}")
drop_list = []
days_n = len(price_df.index)

markets = price_df.columns

for i, f in enumerate(price_df.isna().sum() / days_n):
    if f > 0.9:
        drop_list.append(price_df.columns[i])

print(f"drop length: {len(drop_list)}")

price_df_ = price_df.drop(columns=drop_list)
volume_df_ = volume_df.drop(columns=drop_list)

print(f"remaining length: {len(price_df_.columns)}")

print(f"converting to np array...")

price_m = price_df_.to_numpy(dtype=np.float64).transpose()
volume_m = volume_df_.to_numpy(dtype=np.float64).transpose()

print(f"price_m shape: {price_m.shape}")
print(f"volume_m shape: {volume_m.shape}")

print(f"price_m: {price_m}")

print("start impute price matrix: ")
imputed_price_m = BiScaler(price_m)
# imputed_price_m = knn(price_m)
print("start impute volume matrix: ")
imputed_volume_m = BiScaler(volume_m)
# imputed_volume_m = knn(volume_m)
print("finish imputation")

np.save(f"{path}imputed_price_m.npy", imputed_price_m)
np.save(f"{path}imputed_volume_m.npy", imputed_volume_m)
print("np array saved")

print("start creating new df...")
price_df = pd.DataFrame(index=pd.date_range('2008-1-1', '2018-12-31'),
                        data=imputed_price_m.transpose(), columns=price_df_.columns)
volume_df = pd.DataFrame(index=pd.date_range('2008-1-1', '2018-12-31'),
                         data=imputed_volume_m.transpose(), columns=price_df_.columns)

print("saving to price.csv")
price_df.to_csv(f'{path}brinjal_price.csv')
volume_df.to_csv(f'{path}brinjal_volume.csv')

# init new df
print("init new crop_df")
crop_df = pd.DataFrame(columns={'Date', 'Market', 'Crop', 'Volume', 'Price'}, index=range(0))

# markets list
markets = price_df.columns

print("concatenating two df")
for market in markets:
    df = pd.DataFrame({'Date': pd.date_range("2008-1-1", "2018-12-31"), 'Market': market, 'Crop': crop,
                       'Price': price_df[market], 'Volume': volume_df[market]})
    crop_df = pd.concat([df, crop_df], ignore_index=True)

print("save to brinjal_data.csv")
crop_df.to_csv(f'{path}brinjal_data.csv')
print("Done!")
