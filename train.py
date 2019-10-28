from Train.Dataset import ArrayDataset
from Train.train_epoch import device, train, validate
from Train.utils import norm, norm_arrays
from Model.TCN import TCN, WideTCN
from Model.Wide_Deep_TCN import WideDeepTCN
from DataPreprocessing.PrepareTrainTestArrays import one_hot_encoding

import torch
import torch.nn as nn
import numpy as np
import logging
import os
import time
import sys
import json

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
    print('Loading Data...')
    brinj_price_train_x, brinj_price_train_y, brinj_price_test_x, brinj_price_test_y, \
    brinj_volume_train_x, brinj_volume_test_x, \
    brinj_train_mkt, brinj_test_mkt, brinj_train_geo, brinj_test_geo = \
        load_arrays(path="./np_array/final/4d/Brinjal/",
                    arr_names=["pri_train_x", "pri_train_y", "pri_test_x", "pri_test_y", "vol_train_x", "vol_test_x",\
                               "pri_train_mkt", "pri_test_mkt", "pri_train_geo", "pri_test_geo"])

    brinj_train_crop = np.array([0]*brinj_price_train_x.shape[0]) #one_hot_encoding(np.array([[0]]*price_train_x.shape[0]))
    brinj_test_crop = np.array([0]*brinj_price_test_x.shape[0])#one_hot_encoding(np.array([[0]]*price_test_x.shape[0]))
    
    price_train_x = brinj_price_train_x
    price_train_y = brinj_price_train_y
    price_test_x = brinj_price_test_x
    price_test_y = brinj_price_test_y
    volume_train_x = brinj_volume_train_x
    volume_test_x = brinj_volume_test_x
    train_mkt = brinj_train_mkt
    test_mkt = brinj_test_mkt
    train_geo = brinj_train_geo
    test_geo = brinj_test_geo
    train_crop = brinj_train_crop
    test_crop = brinj_test_crop
    
    print('price_train_x.shape',price_train_x.shape)
    print('price_train_y.shape',price_train_y.shape)
   
    # green chilli

    # tomato

    # 2. concatenate different arrays into one

    # 3. normalize price/volume arrays
    print('Normalizing Data...')
    log.warning("Start Normalizing price training arrays")
    price_train_std, price_train_mean = np.std(price_train_x), np.mean(price_train_x)
    price_train_x, price_test_x = \
        norm_arrays(price_train_x, price_test_x, 
                    log=log, std=price_train_std, mean=price_train_mean)
    
    log.warning("Start Normalizing volume training arrays")
    volume_train_std, volume_train_mean = np.std(volume_train_x), np.mean(volume_train_x)
    volume_train_x, volume_test_x = \
        norm_arrays(volume_train_x, volume_test_x,
                    log=log, std=volume_train_std, mean=volume_train_mean)
    

    # 4. load into dataloaders
    print('Preparing For Dataloader...')
    train_loader = ArrayDataset(price_train_x, price_train_y, volume_train_x, train_mkt, train_geo, train_crop)\
        .data_loader(batch_size=512, shuffle=True, num_workers=1)
    test_loader = ArrayDataset(price_test_x, price_test_y, volume_test_x, test_mkt, test_geo, test_crop)\
        .data_loader(batch_size=512, shuffle=False, num_workers=1)

    # 5. load model
    model = WideDeepTCN(input_len=90, output_len=2, mkt_num=len(train_mkt),
                        mkt_emb=128, crop_num=6, crop_emb=3, num_channels=[16, 8, 4, 1],
                        kernel_size=3, dropout=0.3)
    model.to(device)
    print('Training...')
    # 6. train
#     loss_list, rmse_list, val_loss_list, val_rmse_list = train(model, train_loader=train_loader, val_loader=test_loader,
#           epochs=100, lr=0.01, train_std=price_train_std, train_mean=price_train_mean,\
#           val_std=price_train_std, val_mean=price_train_mean, log=logging, criterion=nn.NLLLoss())
    loss_list, acc_list, val_loss_list, val_acc_list = train(model, train_loader=train_loader, val_loader=test_loader,
          epochs=100, lr=0.001, train_std=price_train_std, train_mean=price_train_mean,\
          val_std=price_train_std, val_mean=price_train_mean, log=logging, criterion=nn.NLLLoss())
    
if __name__ == "__main__":
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
    main()