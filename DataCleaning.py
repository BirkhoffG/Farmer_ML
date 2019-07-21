import pandas as pd
import numpy as np
#%%

def import_df(path, crop):
    price_df = pd.read_csv(f"{path}/{crop}_price.csv", index_col=0, parse_dates=True, low_memory=False)
    volume_df = pd.read_csv(f"{path}/{crop}_volume.csv", index_col=0, parse_dates=True, low_memory=False)
    return price_df, volume_df


crop = 'brinjal'
path = './dataset'

price_df, volume_df = import_df(path, crop)

#%%
# init new df
crop_df = pd.DataFrame(columns={'Date', 'Market', 'Crop', 'Volume', 'Price', 'Label'}, index=range(0))

# markets list
markets = price_df.columns

for market in markets:
    print(f"Start concatenating: {market}")
    label = np.append(price_df[market][1:].to_numpy(),
                      [price_df[market][-1]])

    df = pd.DataFrame({'Date': pd.date_range("2008-1-1", "2018-12-31"), 'Market': market, 'Crop': crop,
                       'Volume': volume_df[market], 'Price': price_df[market], 'Label': label})
    crop_df = pd.concat([df, crop_df], ignore_index=True)

print(crop_df)

#%%

crop_df.to_csv(f"{path}/{crop}_data.csv")


