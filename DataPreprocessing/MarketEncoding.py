import pandas as pd
import numpy as np


class MarketEncoding:
    def __init__(self, path=None):
        if path is None:
            self.market_df = pd.DataFrame(columns=['Market'])
        else:
            self.market_df = pd.read_csv(f"{path}/marketID.csv", index_col=0)

    def add_market(self, market):
        self.market_df = self.market_df.append({'Market': market}, ignore_index=True)

    def get_id(self, market, return_list=True):
        return list(np.where(self.market_df['Market'] == market)[0]) \
            if return_list else list(np.where(self.market_df['Market'] == market)[0])[0]

    def get_market(self, id):
        return self.market_df.iloc[id]['Market']

    def print_df(self):
        print(self.market_df)

    def __len__(self):
        return len(self.market_df)

    def save(self, path):
        try:
            self.market_df.to_csv(f"{path}/marketID.csv")
        except (FileNotFoundError, FileExistsError, NotADirectoryError):
            print(f"No such file: {path}/marketID.csv")
            return False
        return True


m = MarketEncoding()
m.add_market('a')
m.add_market('a')
m.add_market('b')
m.add_market('ac')
m.add_market('a')

print(m.get_id('a'))
print(m.get_id('b', return_list=False))
