from DataPreprocessing.FindLocation import FindLocation as FL
import pandas as pd
import numpy as np


def load_mkts(path, *crops):
    return tuple(pd.read_csv(f'{path}/price_imputed_{crop}.csv',
                             index_col=0, parse_dates=True, low_memory=False).columns.to_list()
                 for crop in crops)


find_location = FL(key='AIzaSyDgI9LZ0Ux2mW-8bEZWkBdOlXMLjZ-P2EE')
brinjal_mkts, tomato_mkts, chilli_mkts = load_mkts('./dataset/data',
                                                   'Brinjal', 'Tomato', 'Green Chilli')

mkts = brinjal_mkts
# ugly code...
for mkt in tomato_mkts:
    if mkt not in mkts:
        mkts.append(mkt)

for mkt in chilli_mkts:
    if mkt not in mkts:
        mkts.append(mkt)
# ugly code end

find_location.find_locations(mkts)
