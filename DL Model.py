from Train.Dataset import ArrayDataset
from Train.train_epoch import device, train
from Train.utils import norm
from Model.TCN import TCN, WideTCN

import torch
import numpy as np
import logging
import os
import time
import sys

# mk dir
data_folder = f'./result/train [{time.strftime("%Y%m%d%H%M%S")}]'
os.mkdir(data_folder)

# set logging configuration
logging.basicConfig(format='[%(asctime)s] - [%(levelname)s] - %(message)s',
                    level=logging.WARN,
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

    # %%


pretrain_path = './np_array/arrays/pretrain/Brinjal/train_90_01_test_90_01/pretrain'

log.info('loading pretrain_x.npy...')
price_pretrain_x = np.load(f'{pretrain_path}/pri_train_x.npy')
volume_pretrain_x = np.load(f'{pretrain_path}/vol_train_x.npy')

log.info('loading pretrain_y.npy...')
price_pretrain_y = np.load(f'{pretrain_path}/pri_train_y.npy')
volume_pretrain_y = np.load(f'{pretrain_path}/vol_train_y.npy')

# %%

training_path = './np_array/arrays/Brinjal/train_90_01_test_90_01/old'
log.info('loading train_x.npy...')
price_train_x = np.load(f'{training_path}/pri_train_x.npy')
volume_train_x = np.load(f'{training_path}/vol_train_x.npy')

log.info('loading train_y.npy...')
price_train_y = np.load(f'{training_path}/pri_train_y.npy')
volume_train_y = np.load(f'{training_path}/vol_train_y.npy')

log.info('loading test_x.npy...')
price_test_x = np.load(f'{training_path}/pri_test_x.npy')
volume_test_x = np.load(f'{training_path}/vol_test_x.npy')

log.info('loading test_y.npy...')
price_test_y = np.load(f'{training_path}/pri_test_y.npy')
volume_test_y = np.load(f'{training_path}/vol_test_y.npy')

# %%
log.warning("Start Normalizing pretraining arrays...")
pre_std_pric, pre_mean_pric = np.std(price_pretrain_x), np.mean(price_pretrain_x)
pre_std_volu, pre_mean_volu = np.std(volume_pretrain_x), np.mean(volume_pretrain_y)
log.info(f"std price = {pre_std_pric}, mean price = {pre_mean_pric}")
log.info(f"std volume = {pre_std_volu}, mean volume = {pre_mean_volu}")

log.info("Normalizing price_pretrain_x...")
price_pretrain_x = norm(price_pretrain_x, pre_std_pric, pre_mean_pric)
stat(price_pretrain_x)
log.info("Normalizing volume_pretrain_x...")
volume_pretrain_x = norm(volume_pretrain_x, pre_std_volu, pre_mean_volu)
stat(volume_pretrain_x)

log.info("Normalizing price_pretrain_y...")
price_pretrain_y = norm(price_pretrain_y, pre_std_pric, pre_mean_pric)
stat(price_pretrain_y)
log.info("Normalizing volume_pretrain_y...")
volume_pretrain_y = norm(volume_pretrain_y, pre_std_volu, pre_mean_volu)
stat(volume_pretrain_y)

# %%
log.warning("Start Normalizing price training arrays")
train_std_pric, train_mean_pric = np.std(price_train_x), np.mean(price_train_x)
log.info(f"std price = {train_std_pric}, mean price = {train_mean_pric}")

log.info("Normalizing price_train_x...")
price_train_x = norm(price_train_x, train_std_pric, train_mean_pric)
stat(price_train_x)
log.info("Normalizing price_train_y...")
price_train_y = norm(price_train_y, train_std_pric, train_mean_pric)
stat(price_train_y)

log.info("Normalizing price_test_x...")
price_test_x = norm(price_test_x, train_std_pric, train_mean_pric)
stat(price_test_x)
log.info("Normalizing price_test_y...")
price_test_y = norm(price_test_y, train_std_pric, train_mean_pric)
stat(price_test_y)

# %%
log.warning("Start Normalizing volume training arrays")
train_std_volu, train_mean_volu = np.std(volume_train_x), np.mean(volume_train_y)
log.info(f"std volume = {train_std_volu}, mean volume = {train_mean_volu}")

log.info("Normalizing volume_train_x...")
volume_train_x = norm(volume_train_x, train_std_volu, train_mean_volu)
stat(volume_train_x)
log.info("Normalizing volume_train_y...")
volume_train_y = norm(volume_train_y, train_std_volu, train_mean_volu)
stat(volume_train_y)

log.info("Normalizing volume_test_x...")
volume_test_x = norm(volume_test_x, train_std_volu, train_mean_volu)
stat(volume_test_x)
log.info("Normalizing volume_test_y...")
volume_test_y = norm(volume_test_y, train_std_volu, train_mean_volu)
stat(volume_test_y)

# %%

print("*" * 36)

price_pretrain_x = np.expand_dims(price_pretrain_x, axis=-1)
log.info(f"price_pretrain_x shape: {price_pretrain_x.shape}")
log.info(f"price_pretrain_y shape: {price_pretrain_y.shape}")

volume_pretrain_x = np.expand_dims(volume_pretrain_x, axis=-1)
log.info(f"volume_pretrain_x shape: {volume_pretrain_x.shape}")
log.info(f"volume_pretrain_y shape: {volume_pretrain_y.shape}")

print("*" * 36)

train_x = np.stack((price_train_x, volume_train_x), axis=-1)
train_y = np.copy(price_train_y)
log.info(f"train_x shape: {train_x.shape}")
log.info(f"train_y shape: {train_y.shape}")

test_x = np.stack((price_test_x, volume_test_x), axis=-1)
test_y = np.copy(price_test_y)
log.info(f"test_x shape: {test_x.shape}")
log.info(f"test_y shape: {test_y.shape}")

# %%
log.warning("Prepare to load pretrain arrays.")
p_batch_size = 256
log.info(f"batch size = {p_batch_size}; shuffle = True (Default); num of workers = 4 (Default)")

log.info(f"loading pretrain price arrays...")
price_pretrain_loader = ArrayDataset(price_pretrain_x, price_pretrain_y).data_loader(batch_size=p_batch_size)
log.info(f"loading pretrain volume arrays...")
volume_pretrain_loader = ArrayDataset(volume_pretrain_x, volume_pretrain_y).data_loader(batch_size=p_batch_size)

# %%

log.warning("Prepare to load training arrays.")
t_batch_size = 256
log.info(f"batch size = {t_batch_size}; shuffle = True (Default); num of workers = 4 (Default)")

log.info(f"loading pretrain price arrays...")
train_loader = ArrayDataset(train_x, train_y).data_loader(batch_size=t_batch_size)
log.info(f"loading pretrain volume arrays...")
test_loader = ArrayDataset(test_x, test_y).data_loader(batch_size=t_batch_size)

# %%

log.warning("Loading price pretraining model...")
p_model = TCN(input_size=1, output_size=1, num_channels=[16, 8, 4, 2, 1], input_len=90, output_len=1,
              kernel_size=3, dropout=0.3, feature='pri')
p_model.to(device)

log.info("Start price pretraining...")
p_loss_list, p_rmse_list, p_val_loss_list, p_val_rmse_list = \
    train(train_loader=price_pretrain_loader, val_loader=test_loader, model=p_model, epochs=50, lr=0.01,
          train_std=pre_std_pric, train_mean=pre_mean_pric, val_std=train_std_pric, val_mean=train_mean_pric, log=log)

torch.save(p_model.state_dict(), f'{data_folder}/p_model.pt')
save_arrays(p_loss_list=p_loss_list, p_rmse_list=p_rmse_list,
            p_val_loss_list=p_val_loss_list, p_val_rmse_list=p_val_rmse_list)

# %%

log.warning("Loading volume pretraining model...")
v_model = TCN(input_size=1, output_size=1, num_channels=[16, 8, 4, 2, 1], input_len=90, output_len=1,
              kernel_size=3, dropout=0.3, feature='vol')
v_model.to(device)

log.info("Start volume pretraining...")
v_loss_list, v_rmse_list, v_val_loss_list, v_val_rmse_list = \
    train(train_loader=volume_pretrain_loader, val_loader=test_loader, model=v_model, epochs=50, lr=0.01,
          train_std=pre_std_volu, train_mean=pre_mean_volu, val_std=train_std_volu, val_mean=train_mean_volu, log=log)

torch.save(v_model, f'{data_folder}/v_model.pt')
save_arrays(v_loss_list=v_loss_list, v_rmse_list=v_rmse_list,
            v_val_loss_list=v_val_loss_list, v_val_rmse_list=v_val_rmse_list)

# %%
log.warning("Loading WideTCN model...")
train_model = WideTCN(input_size=2, output_size=1, num_channels=[16, 8, 4, 2, 1], input_len=90, output_len=1,
                      kernel_size=3, dropout=0.3, p_tcn=p_model.tcn, v_tcn=v_model.tcn)
train_model.to(device)

log.info("Start training...")
loss_li, rmse_li, val_loss_li, val_rmse_li = \
    train(train_loader=train_loader, val_loader=test_loader, model=train_model, epochs=30, lr=0.01,
          train_std=train_std_pric, train_mean=train_mean_pric, val_std=train_std_pric, val_mean=train_mean_pric,
          log=log)

torch.save(train_model, f'{data_folder}/train_model.pt')
save_arrays(loss_li=loss_li, rmse_li=rmse_li, val_loss_li=val_loss_li, val_rmse_li=val_rmse_li)

# %%
log.warning("Loading non-pretrain WideTCN model...")
nap_model = WideTCN(input_size=2, output_size=1, num_channels=[16, 8, 4, 2, 1], input_len=90, output_len=1,
                    kernel_size=3, dropout=0.3, p_tcn=None, v_tcn=None)
nap_model.to(device)

log.info("Start training...")
nap_loss_li, nap_rmse_li, nap_val_loss_li, nap_val_rmse_li = \
    train(train_loader=train_loader, val_loader=test_loader, model=nap_model, epochs=50, lr=0.01,
          train_std=train_std_pric, train_mean=train_mean_pric, val_std=train_std_pric, val_mean=train_mean_pric,
          log=log)

torch.save(nap_model, f'{data_folder}/nap_model.pt')
save_arrays(nap_loss_li=nap_loss_li, nap_rmse_li=nap_rmse_li,
            nap_val_loss_li=nap_val_loss_li, nap_val_rmse_li=nap_val_rmse_li)

log.info("Done.")
