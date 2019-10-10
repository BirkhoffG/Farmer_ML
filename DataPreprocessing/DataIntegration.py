import pandas as pd
import os
import io
from datetime import datetime
import numpy as np

from DataPreprocessing import CROPS, STATES


class IntegrateDataFrame(object):
    def __init__(self, data_path, crop, export_path="./dataset",
                 date_start="2008-1-1", date_end="2018-12-31", reindex=False):
        self.data_path = data_path
        if crop in CROPS:
            self.crop = crop
        else:
            raise ValueError(f"No such crop: {crop}")
        self.export_path = export_path
        self.price_df = pd.DataFrame(index=pd.date_range(date_start, date_end))
        self.volume_df = pd.DataFrame(index=pd.date_range(date_start, date_end))
        self.max_df = pd.DataFrame(index=pd.date_range(date_start, date_end))
        self.min_df = pd.DataFrame(index=pd.date_range(date_start, date_end))
        print(f"Start loading {crop} data:")
        self.load_data(data_path)
        if reindex:
            self.reindex()
        print("Done.")

    def load_file(self, file):
        state = file.split('_')[2]
        df = pd.read_csv(f'./{self.data_path}/{file}')
        markets = df['Market'].unique()

        for market in markets:
            # set market df into datetime index
            market_df = df.iloc[list(np.where(df['Market'] == market)[0])]
            market_df = market_df.dropna(subset=['Arrival Date'])
            market_df.loc[:, 'Arrival Date'] = pd.to_datetime(market_df['Arrival Date'], format='%d/%m/%Y')
            market_df.set_index('Arrival Date', inplace=True)
            # name column
            market = f'{market}_{state}'

            # append market column if not exits
            if market not in self.price_df.columns:
                self.price_df[market] = np.nan
            if market not in self.volume_df.columns:
                self.volume_df[market] = np.nan
            if market not in self.max_df.columns:
                self.max_df[market] = np.nan
            if market not in self.min_df.columns:
                self.min_df[market] = np.nan

            # resample market df in daily bases
            market_df = market_df.resample('D').asfreq()

            try:
                self.price_df[market].loc[market_df.index[0]: market_df.index[-1]] = \
                    market_df['Modal Price(Rs./Quintal)'].to_numpy()
                self.volume_df[market].loc[market_df.index[0]: market_df.index[-1]] = \
                    market_df['Arrivals (Tonnes)'].to_numpy()
                self.min_df[market].loc[market_df.index[0]: market_df.index[-1]] = \
                    market_df['Minimum Price(Rs./Quintal)'].to_numpy()
                self.max_df[market].loc[market_df.index[0]: market_df.index[-1]] = \
                    market_df['Maximum Price(Rs./Quintal)'].to_numpy()
            except ValueError:
                print(f"Error Market: {market}")
                print(f"time length: {market_df.index[0]}: {market_df.index[-1]}")
                continue

    def load_data(self, path):
        file_list = os.listdir(path)

        for ix, file in enumerate(file_list):
            # check the crop type
            if file.split('_')[3] != f'{self.crop[:4]}.csv':
                continue
            print(f"No.{ix}: {file}")

            self.load_file(f"{file}")

    def reindex(self):
        """Cluster all market in the same state together"""

        headers = self.price_df.columns
        assert headers == volume_df.columns
        assert headers == max_df.columns
        assert headers == min_df.columns

        new_headers = []
        market_dict = {state[:4]: [] for state in STATES}
        for head in headers:
            market, state = head.split("_")
            market_dict[state].append(head)
        for _, head in market_dict.items():
            new_headers.extend(head)
        self.price_df = self.price_df.reindex(columns=new_headers)
        self.volume_df = self.volume_df.reindex(columns=new_headers)
        self.max_df = self.max_df.reindex(columns=new_headers)
        self.min_df = self.min_df.reindex(columns=new_headers)

    def export(self):
        return self.price_df, self.volume_df, self.max_df, self.min_df


if __name__ == '__main__':

    for crop in CROPS:
        path = f'../Raw Data/{crop}'

        print(f"Concatenating {crop}")
        price_df, volume_df, min_df, max_df = IntegrateDataFrame(data_path=path, crop=crop).export()

        print(f"storing ../dataset/price_{crop}.csv...")
        price_df.Fto_csv(f"../dataset/price_{crop}.csv")

        print(f"storing ../dataset/volume_{crop}.csv...")
        volume_df.to_csv(f"../dataset/volume_{crop}.csv")

        print(f"storing ../dataset/min_price_{crop}.csv...")
        min_df.to_csv(f"../dataset/min_price_{crop}.csv")

        print(f"storing ../dataset/max_price_{crop}.csv...")
        max_df.to_csv(f"../dataset/max_price_{crop}.csv")

