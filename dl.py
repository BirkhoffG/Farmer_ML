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
    # pretrain path
    p_path = param.get('p_path')
    # train path
    t_path = param.get('t_path')
    # pretrain dataloader's batch size
    p_batch_size = param.get('p_batch_size')
    # train dataloader's batch size
    t_batch_size = param.get('t_batch_size')
    # price pretrain parameters
    p_pretrain = param.get('price_pretrain')
    # whether do price pretrain
    do_price_pretrain = p_pretrain.get('do_pretrain')
    # price pretrain model's parameters
    p_model_param = p_pretrain.get('model')
    # price pretrain parameters
    p_pretrain_param = p_pretrain.get('train')
    # volueme pretrain parameters
    v_pretrain = param.get('volume_pretrain')
    # whether do volume pretrain
    do_volume_pretrain = v_pretrain.get('do_pretrain')
    # volume pretrain model's parameters
    v_model_param = v_pretrain.get('model')
    # volume pretrain parameters
    v_pretrain_param = v_pretrain.get('train')

    log.info('loading pretrain_x.npy...')
    price_pretrain_x = np.load(f'{p_path}/pri_train_x.npy')
    volume_pretrain_x = np.load(f'{p_path}/vol_train_x.npy')

    log.info('loading pretrain_y.npy...')
    price_pretrain_y = np.load(f'{p_path}/pri_train_y.npy')
    volume_pretrain_y = np.load(f'{p_path}/vol_train_y.npy')

    log.info('loading train_x.npy...')
    price_train_x = np.load(f'{t_path}/pri_train_x.npy')
    volume_train_x = np.load(f'{t_path}/vol_train_x.npy')

    log.info('loading train_y.npy...')
    price_train_y = np.load(f'{t_path}/pri_train_y.npy')
    volume_train_y = np.load(f'{t_path}/vol_train_y.npy')

    log.info('loading test_x.npy...')
    price_test_x = np.load(f'{t_path}/pri_test_x.npy')
    volume_test_x = np.load(f'{t_path}/vol_test_x.npy')

    log.info('loading test_y.npy...')
    price_test_y = np.load(f'{t_path}/pri_test_y.npy')
    volume_test_y = np.load(f'{t_path}/vol_test_y.npy')

    log.warning("Start Normalizing pretraining arrays...")
    log.info("Normalizing price_pretrain_x...")
    price_pretrain_x, price_pretrain_std, price_pretrain_mean = norm(arr=price_pretrain_x, log=log)
    stat(price_pretrain_x)
    log.info("Normalizing price_pretrain_y...")
    price_pretrain_y, _, _ = norm(arr=price_pretrain_y, log=log, std=price_pretrain_std, mean=price_pretrain_mean)
    stat(price_pretrain_y)

    log.info("Normalizing volume_pretrain_x...")
    volume_pretrain_x, volume_pretrain_std, volume_pretrain_mean = norm(arr=volume_pretrain_x, log=log)
    stat(volume_pretrain_x)
    log.info("Normalizing volume_pretrain_y...")
    volume_pretrain_y, _, _ = norm(arr=volume_pretrain_y, log=log, std=volume_pretrain_std, mean=volume_pretrain_mean)
    stat(volume_pretrain_y)

    log.warning("Start Normalizing price training arrays")
    price_train_std, price_train_mean = np.std(price_train_x), np.mean(price_train_x)
    price_train_x, price_train_y, price_test_x, price_test_y = \
        norm_arrays(price_train_x, price_train_y, price_test_x, price_test_y,
                    log=log, std=price_train_std, mean=price_train_mean)

    # print statistic metrics for price arrays
    log.info("Normalizing price_train_x..."); stat(price_train_x)
    log.info("Normalizing price_train_y..."); stat(price_train_y)
    log.info("Normalizing price_test_x..."); stat(price_test_x)
    log.info("Normalizing price_test_y..."); stat(price_test_y)

    log.warning("Start Normalizing volume training arrays")
    volume_train_std, volume_train_mean = np.std(volume_train_x), np.mean(volume_train_y)
    volume_train_x, volume_train_y, volume_test_x, volume_test_y = \
        norm_arrays(volume_train_x, volume_train_y, volume_test_x, volume_test_y,
                    log=log, std=volume_train_std, mean=volume_train_mean)

    # print statistic metrics for price arrays
    log.info("Normalizing volume_train_x..."); stat(volume_train_x)
    log.info("Normalizing volume_train_y..."); stat(volume_train_y)
    log.info("Normalizing volume_test_x..."); stat(volume_test_x)
    log.info("Normalizing volume_test_y..."); stat(volume_test_y)

    price_pretrain_x = np.expand_dims(price_pretrain_x, axis=-1)
    log.info(f"price_pretrain_x shape: {price_pretrain_x.shape}")
    log.info(f"price_pretrain_y shape: {price_pretrain_y.shape}")

    volume_pretrain_x = np.expand_dims(volume_pretrain_x, axis=-1)
    log.info(f"volume_pretrain_x shape: {volume_pretrain_x.shape}")
    log.info(f"volume_pretrain_y shape: {volume_pretrain_y.shape}")

    train_x = np.stack((price_train_x, volume_train_x), axis=-1)
    train_y = np.copy(price_train_y)
    log.info(f"train_x shape: {train_x.shape}")
    log.info(f"train_y shape: {train_y.shape}")

    test_x = np.stack((price_test_x, volume_test_x), axis=-1)
    test_y = np.copy(price_test_y)
    log.info(f"test_x shape: {test_x.shape}")
    log.info(f"test_y shape: {test_y.shape}")

    log.info(f"loading pretrain price arrays...")
    price_pretrain_loader = ArrayDataset(price_pretrain_x, price_pretrain_y).data_loader(batch_size=p_batch_size)
    log.info(f"loading pretrain volume arrays...")
    volume_pretrain_loader = ArrayDataset(volume_pretrain_x, volume_pretrain_y).data_loader(batch_size=p_batch_size)

    log.info(f"loading train arrays...")
    train_loader = ArrayDataset(train_x, train_y).data_loader(batch_size=t_batch_size)
    log.info(f"loading validate arrays...")
    test_loader = ArrayDataset(test_x, test_y).data_loader(batch_size=t_batch_size)

    if do_price_pretrain:
        log.warning("Loading price pretraining model...")
        p_model = TCN(**p_model_param)
        p_model.to(device)

        log.info("Start price pretraining...")
        p_loss_list, p_rmse_list, p_val_loss_list, p_val_rmse_list = \
            train(**p_pretrain_param, train_loader=price_pretrain_loader, val_loader=test_loader, model=p_model,
                  train_std=price_pretrain_std, train_mean=price_pretrain_mean,
                  val_std=price_train_std, val_mean=price_train_mean, log=log)

        torch.save(p_model.state_dict(), f'{data_folder}/p_model.pt')
        save_arrays(p_loss_list=p_loss_list, p_rmse_list=p_rmse_list,
                    p_val_loss_list=p_val_loss_list, p_val_rmse_list=p_val_rmse_list)

    if do_volume_pretrain:
        log.warning("Loading volume pretraining model...")
        v_model = TCN(**v_model_param)
        v_model.to(device)
        print(device)

        log.info("Start volume pretraining...")
        v_loss_list, v_rmse_list, v_val_loss_list, v_val_rmse_list = \
            train(**v_pretrain_param, train_loader=volume_pretrain_loader, val_loader=test_loader, model=v_model,
                  train_std=volume_pretrain_std, train_mean=volume_pretrain_mean,
                  val_std=volume_train_std, val_mean=volume_train_mean, log=log)

        torch.save(v_model, f'{data_folder}/v_model.pt')
        save_arrays(v_loss_list=v_loss_list, v_rmse_list=v_rmse_list,
                    v_val_loss_list=v_val_loss_list, v_val_rmse_list=v_val_rmse_list)


if __name__ == '__main__':
    param = {
        'p_path': './np_array/arrays/pretrain/Brinjal/train_90_01_test_90_01/pretrain',
        't_path': './np_array/arrays/Brinjal/train_90_01_test_90_01/old',
        'p_batch_size': 256,
        't_batch_size': 256,
        'price_pretrain': {
            'do_pretrain': True,
            'model': {
                'input_size': 1,
                'output_size': 1,
                'num_channels': [16, 8, 4, 2, 1],
                'input_len': 90,
                'output_len': 1,
                'kernel_size': 3,
                'dropout': 0.3,
                'feature': 'pri'
            },
            'train': {
                'epochs': 50,
                'lr': 0.01
            }
        },
        'volume_pretrain': {
            'do_pretrain': True,
            'model': {
                'input_size': 1,
                'output_size': 1,
                'num_channels': [16, 8, 4, 2, 1],
                'input_len': 90,
                'output_len': 1,
                'kernel_size': 3,
                'dropout': 0.3,
                'feature': 'vol'
            },
            'train': {
                'epochs': 50,
                'lr': 0.01
            }
        }
    }
    main(**param)
