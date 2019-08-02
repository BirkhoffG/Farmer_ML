import pandas as pd
import os
import io
from datetime import datetime
import numpy as np


class ConCatenateDataFrame(object):
    def __init__(self, data_path, crop, export_path="./concatenated_dataset",
                 date_start="2008-1-1", date_end="2018-12-31", reindex=False):
        self.data_path = data_path
        if crop in ['Brinjal', 'Tomato', 'Mango', 'Cauliflower', 'Pointed gourd (Parval)', 'Green Chilli']:
            self.crop = crop
        else:
            raise ValueError(f"No such crop: {crop}")
        self.export_path = export_path
        self.price_df = pd.DataFrame(index=pd.date_range(date_start, date_end))
        self.volume_df = pd.DataFrame(index=pd.date_range(date_start, date_end))
        print(f"Start loading {crop} data:")
        self.load_data(data_path)
        if reindex:
            self.reindex()
        print("Done.")
        # self.load_file("May_08_Goa_Mang.csv")
        # price_df = self.price_df
        # volume_df = self.volume_df
        # print(self.price_df)
        # print(self.volume_df.count())

    def load_file(self, file):
        # state
        state = file.split('_')[2]
        df = pd.read_csv(f'./{self.data_path}/{file}')
        markets = df['Market'].unique()

        for market in markets:
            # set market df into datetime index
            market_df = df.iloc[list(np.where(df['Market'] == market)[0])]
            market_df = market_df.dropna(subset=['Arrival Date'])
            market_df.loc[:, 'Arrival Date'] = pd.to_datetime(market_df['Arrival Date'], format='%d/%m/%Y')
            market_df.set_index('Arrival Date', inplace=True)

            market = f'{market}_{state}'
            # print(f'market: {market}')

            # append market column if not exits
            if market not in self.price_df.columns:
                self.price_df[market] = np.nan
            if market not in self.volume_df.columns:
                self.volume_df[market] = np.nan

            # resample market df
            market_df = market_df.resample('D').asfreq()
            # print(market_df)
            # print(f'{price_df[market].loc[market_df.index[0]: market_df.index[-1]]}')
            # print(f"{market_df['Modal Price(Rs./Quintal)']}")
            try:
                # self.price_df[market].loc[market_df.index[0]: market_df.index[-1]] = \
                #     market_df['Modal Price(Rs./Quintal)'].to_numpy()
                # self.volume_df[market].loc[market_df.index[0]: market_df.index[-1]] = \
                #     market_df['Arrivals (Tonnes)'].to_numpy()
                self.price_df[market].loc[market_df.index[0]: market_df.index[-1]] = \
                    market_df['Minimum Price(Rs./Quintal)'].to_numpy()
                self.volume_df[market].loc[market_df.index[0]: market_df.index[-1]] = \
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
            # if file == start_point:
            #     start_concate = True
            #     continue
            # if not start_concate:
            #     continue
            self.load_file(f"{file}")

    def reindex(self):
        """Cluster all market in the same state together"""
        states = ['Andaman and Nicobar', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh',
                  'Chattisgarh', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Goa', 'Gujarat', 'Haryana',
                  'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Lakshadweep',
                  'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'NCT of Delhi',
                  'Odisha', 'Pondicherry', 'Punjab', 'Rajasthan', 'Tamil Nadu', 'Telangana', 'Tripura',
                  'Uttar Pradesh', 'Uttrakhand', 'West Bengal']

        headers = self.price_df.columns
        # TODO.md check if price_df.columns == volume_df.columns
        new_headers = []
        market_dict = {state[:4]: [] for state in states}
        for head in headers:
            market, state = head.split("_")
            market_dict[state].append(head)
        for _, head in market_dict.items():
            new_headers.extend(head)
        self.price_df = self.price_df.reindex(columns=new_headers)
        self.volume_df = self.volume_df.reindex(columns=new_headers)

    def export(self):
        return self.price_df, self.volume_df


if __name__ == '__main__':
    crop = 'Brinjal'
    path = '../Raw Data/Brinjal'

    min_df, max_df = ConCatenateDataFrame(data_path=path, crop=crop).export()

    print("storing price df...")
    min_df.to_csv(f"../dataset/min_price_{crop}.csv")

    print("storing volume df...")
    max_df.to_csv(f"../dataset/max_price_{crop}.csv")

