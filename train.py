from Train.Dataset import ArrayDataset
from Train.train_epoch import device, train, validate
from Train.utils import norm, norm_arrays
from Model.TCN import TCN, WideTCN

import torch
import numpy as np
import logging
import os
import time
import sys
import json

# mk dir
data_folder = f'./result/train [{time.strftime("%Y-%m-%d-%H-%M-%S")}]'
os.mkdir(data_folder)

# set logging configuration
logging.basicConfig(format='[%(asctime)s] - [%(levelname)s] - %(message)s',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{data_folder}/logs.log'),
                        logging.StreamHandler(sys.stdout)
                    ])
# get logger
log = logging.getLogger(__name__)
log.warning('This will get logged to a file')


def stat(arr):
    log.info(f"std: {np.std(arr)}; mean: {np.mean(arr)}")


def save_arrays(**arrs):
    for name, li in arrs.items():
        log.info(f"Saving {data_folder}/{name}.npy")
        np.save(f'{data_folder}/{name}.npy', np.array(li))


def main(**param):
    return
