import netCDF4
import numpy as np
import pandas as pd

dataset = netCDF4.Dataset('../weather/Complete_TMAX_Daily_LatLong1_2000.nc')

#%%

# variables

# np longitude array: (longitude: 360)
lon = dataset['longitude'][:]
# np latitude array: (latitude: 180)
lat = dataset['latitude'][:]
# np temperature array: (time: 3653, latitude: 180, longitude: 360)
temperature = dataset['temperature'][:]

#%%


def get_closest_lat_lon(lat_loc, lon_loc):
    """India is situated north of the equator
    between 8°4' to 37°6' north latitude
    and 68°7' to 97°25' east longitude
    with a total area of 3,287,263 square kilometres

    :param lat_loc latitude point
    :param lon_loc longitude point
    """
    return int(lat_loc) + 0.5, int(lon_loc) + 0.5


def get_loc_temp(lat_loc, lon_loc, date: pd.datetime):
    lat_loc, lon_loc = get_closest_lat_lon(lat_loc, lon_loc)
    assert 8.4 < lat < 37.6
    assert 68.7 < lon < 97.25

    lat_ix = lat_loc - 0.5 + 90; assert lat[lat_ix] == lat_loc
    lon_ix = lon_loc - 0.5 + 90; assert lon[lon_ix] == lon_loc

    d = (date - pd.to_datetime('2000-1-01')).days
    return temperature[lat + 90, lon + 180, d]
