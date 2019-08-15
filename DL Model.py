from Train.Dataset import *
from Train.utils import *
from Train.train_epoch import *
from Model.TCN import TCN

import logging as log

# logging.basicConfig(filename='DL Model.log', filemode='w', format='%(asctime)s - %(message)s')
# logging.warning('This will get logged to a file')

#%%

pretrain_path = './np_array/arrays/pretrain/Brinjal/train_90_01_test_90_01/pretrain'

log.info('loading pretrain_x.npy...')
price_pretrain_x = np.load(f'{pretrain_path}/pri_train_x.npy')
volume_pretrain_x = np.load(f'{pretrain_path}/vol_train_x.npy')

log.info('loading pretrain_y.npy...')
price_pretrain_y = np.load(f'{pretrain_path}/pri_train_y.npy')
volume_pretrain_y = np.load(f'{pretrain_path}/vol_train_y.npy')


#%%

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

#%%
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

#%%
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

#%%
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

#%%

print("*"*36)

price_pretrain_x = np.expand_dims(price_pretrain_x, axis=-1)
log.info(f"price_pretrain_x shape: {price_pretrain_x.shape}")
log.info(f"price_pretrain_y shape: {price_pretrain_y.shape}")

volume_pretrain_x = np.expand_dims(volume_pretrain_x, axis=-1)
log.info(f"volume_pretrain_x shape: {volume_pretrain_x.shape}")
log.info(f"volume_pretrain_y shape: {volume_pretrain_y.shape}")

print("*"*36)

train_x = np.stack((price_train_x, volume_train_x), axis=-1)
train_y = np.copy(price_train_y)
log.info(f"train_x shape: {train_x.shape}")
log.info(f"train_y shape: {train_y.shape}")

test_x = np.stack((price_test_x, volume_test_x), axis=-1)
test_y = np.copy(price_test_y)
log.info(f"test_x shape: {test_x.shape}")
log.info(f"test_y shape: {test_y.shape}")

#%%
log.warning("Prepare to load pretrain arrays.")
pretrain_batch_size = 128
log.info(f"batch size = {pretrain_batch_size}; shuffle = True (Default); num of workers = 4 (Default)")

log.info(f"loading pretrain price arrays...")
price_pretrain_loader = ArrayDataset(price_pretrain_x, price_pretrain_y).data_loader(batch_size=pretrain_batch_size)
log.info(f"loading pretrain volume arrays...")
volume_pretrain_loader = ArrayDataset(volume_pretrain_x, volume_pretrain_y).data_loader(batch_size=pretrain_batch_size)

#%%

log.warning("Prepare to load training arrays.")
training_batch_size = 256
log.info(f"batch size = {training_batch_size}; shuffle = True (Default); num of workers = 4 (Default)")

log.info(f"loading pretrain price arrays...")
train_loader = ArrayDataset(train_x, train_y).data_loader(batch_size=training_batch_size)
log.info(f"loading pretrain volume arrays...")
test_loader = ArrayDataset(test_x, test_y).data_loader(batch_size=training_batch_size)

#%%

log.warning("Loading price pretraining model...")
p_model = TCN(input_size=1, output_size=1, num_channels=[16, 8, 4, 1], input_len=90, output_len=1,
                 kernel_size=3, dropout=0.3, emb_dropout=0.1, feature='pri')
p_model.to(device)
print(p_model)

log.info("Start training...")
loss_list, rmse_list, val_loss_list, val_rmse_list = \
    train(train_loader=price_pretrain_loader, val_loader=test_loader, model=p_model, epochs=30, lr=0.01,
         train_std=pre_std_pric, train_mean=pre_mean_pric, val_std=train_std_pric, val_mean=train_mean_pric)
