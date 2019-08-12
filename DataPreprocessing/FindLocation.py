import json
import googlemaps
from DataPreprocessing import STATES


class FindLocation:
    def __init__(self, key, mkt2loc_path=None):
        self.state_abbr = {state[:4]: state for state in STATES}
        self.gmaps = googlemaps.Client(key=key)

        if mkt2loc_path is not None:
            with open(mkt2loc_path, 'r') as file:
                self.mkt2loc = json.load(file)
        else:
            self.mkt2loc = dict()
        self.exception_list = []

    def find_location(self, market, query=None):
        mkt, state = market.split('_')
        if query is None:
            query = f"{self.state_abbr[state]} {mkt}"
        try:
            place = self.gmaps.places(query=query)
            loc = place['results'][0]['geometry']['location']
            self.mkt2loc.update({market: loc})
            print(f"{market}: {loc}")
        except IndexError:
            self.exception_list.append(market)
            print(f"Error: {market}")

    def find_locations(self, markets):
        for market in markets:
            self.find_location(market)

    def save_dict(self, path):
        with open(f'{path}/mkt2loc.json', 'w') as fp:
            json.dump(self.mkt2loc, fp)

    def print_error_list(self):
        print(self.exception_list)
        return self.exception_list
