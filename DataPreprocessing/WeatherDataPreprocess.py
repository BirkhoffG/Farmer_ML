import netCDF4
import numpy as np

dataset = netCDF4.Dataset('../weather/Complete_TMAX_Daily_LatLong1_2000.nc')

#%%

# variables

# np longitude array
lon = dataset.variables['longitude'][:]
# np latitude array
lat = dataset.variables['latitude'][:]
# np temperature array
temperature = dataset.variables['temperature'][:]

#%%


def getclosest_ij(lats, lons, latpt, lonpt):
    # find squared distance of every point on grid
    dist_sq = (lats-latpt)**2 + (lons-lonpt)**2
    # 1D index of minimum dist_sq element
    minindex_flattened = dist_sq.argmin()
    # Get 2D index for latvals and lonvals arrays from 1D index
    return np.unravel_index(minindex_flattened, lats.shape)

