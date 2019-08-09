import ee
import ssl
import time
from pathlib import Path
import numpy as np

collection_id = "IDAHO_EPSCOR/TERRACLIMATE"
ee.Initialize()

img_col = ee.ImageCollection(collection_id) \
    .filterBounds(ee.Geometry.Rectangle(68.7, 97.2, 8.4, 37.6)) \
    .filterDate('2008-01-01', '2018-12-31')

bands = img_col.select(['tmmx', 'tmmn', 'vs', 'pr'])

img = ee.Image(img_col)
print(img)
