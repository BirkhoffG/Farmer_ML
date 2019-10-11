from Train.Dataset import ArrayDataset
from Train.train_epoch import device, train, validate
from Train.utils import norm, norm_arrays
from Model.TCN import TCN, WideTCN
from Model.Wide_Deep_TCN import WideDeepTCN

import torch
import torch.nn as nn
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


def load_arrays(path:str, arr_names:list):
    return tuple(np.load(f'{path}/{name}.npy') for name in arr_names)


def main(**param):
    # 1. load all arrays
    # TODO: one-hot-encoding crop arrays
    # brinjal arrays
    brinj_price_train_x, brinj_price_train_y, brinj_price_test_x, brinj_price_test_y, \
    brinj_volume_train_x, brinj_volume_train_y, brinj_volume_test_x, brinj_volume_test_y, \
    brinj_train_mkt, brinj_test_mkt, brinj_train_geo, brinj_test_geo = \
        load_arrays(path="./np_array/arrays/Brinjal/train_90_01_test_90_01",
                    arr_names=["pri_train_x", "pri_train_y"])

    # green chilli

    # tomato

    # 2. concatenate different arrays into one

    # 3. normalize price/volume arrays

    # 4. load into dataloaders
    train_loader = ArrayDataset(brinj_price_train_x, brinj_volume_train_x, brinj_price_train_y)\
        .data_loader(batch_size=512, shuffle=True, num_workers=1)
    test_loader = ArrayDataset(brinj_price_test_x, brinj_volume_test_x, brinj_price_test_y)\
        .data_loader(batch_size=512, shuffle=False, num_workers=1)

    # 5. load model
    model = WideDeepTCN(input_len=90, output_len=1, mkt_num=len(brinj_train_mkt),
                        mkt_emb=128, crop_num=6, crop_emb=3, num_channels=[16, 8, 4, 1],
                        kernel_size=3, dropout=0.3)
    model.to(device)

    # 6. train
    loss_list, rmse_list, val_loss_list, val_rmse_list = train(model, train_loader=train_loader, val_loader=test_loader,
          epochs=100, lr=0.01, criterion=nn.L1Loss(), )