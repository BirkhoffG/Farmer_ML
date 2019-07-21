import json
import pandas as pd
import googlemaps
import pprint

#%%

states = ['Andaman and Nicobar', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh',
                  'Chattisgarh', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Goa', 'Gujarat', 'Haryana',
                  'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Lakshadweep',
                  'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'NCT of Delhi',
                  'Odisha', 'Pondicherry', 'Punjab', 'Rajasthan', 'Tamil Nadu', 'Telangana', 'Tripura',
                  'Uttar Pradesh', 'Uttrakhand', 'West Bengal']
state_abbr = {state[:4]: state for state in states}

markets = pd.read_csv('./price_brinjal.csv', index_col=0, parse_dates=True, low_memory=False).columns

gmaps = googlemaps.Client(key='AIzaSyDgI9LZ0Ux2mW-8bEZWkBdOlXMLjZ-P2EE')

# queries = [f"{state_abbr[market.split('_')[1]]}_{market.split('_')[0]}" for market in markets]
mkt2loc = dict()

exception_list = []

for market in markets:
    mkt, state = market.split('_')
    query = f"{state_abbr[state]} {mkt}"
    try:
        place = gmaps.places(query=query)
        loc = place['results'][0]['geometry']['location']
        mkt2loc.update({market: loc})
        print(f"{market}: {loc}")
    except IndexError:
        exception_list.append(market)
        print(f"Error: {market}")

with open('mkt2loc.json', 'w') as fp:
    json.dump(mkt2loc, fp)

#%%

