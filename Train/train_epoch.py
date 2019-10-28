import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import time
import numpy as np
import logging
from Train.utils import device, list2arr, val_RMSE, RMSE

from tqdm import tqdm

def cal_acc(pred, gold):
    gold = gold.long()
    length = len(gold)
    pred = pred.max(1)[1]
    n_correct = pred.eq(gold).sum().item()
    return n_correct/length
    
def validate(model: nn.Module, val_loader: DataLoader):
    """
    validate the test set in the model
    :param model: neural network modal
    :param val_loader: test loader
    :param std: standard deviation
    :param mean: mean value
    :return: RMSE on validation set; loss list on validation set
    """
    # track values
    pred_list, acc_list, loss_list = [], [], []
    # L2 loss
    criterion = nn.NLLLoss()
    # model evaluation
    model.eval()
    with torch.no_grad():
        for i, (x, y, x_vol, x_mkt, x_geo, x_crop) in enumerate(val_loader):
            y = y.view(-1)
            x, y, x_vol, x_geo, x_mkt, x_crop = x.float(), y.float(), x_vol.float(), x_geo.float(), x_mkt.long(), x_crop.long()
            # prepare the data and label
            x, y, x_vol, x_geo, x_mkt, x_crop = x.to(device), y.to(device), x_vol.to(device),\
                                                x_geo.to(device), x_mkt.to(device), x_crop.to(device)
            
            result = model(x, x_vol, x_geo, x_mkt, x_crop)
            acc = cal_acc(result,y)
            loss = criterion(result, y.long())
            
            acc_list.append(acc)
            loss_list.append(loss.item())
            
    return acc_list,loss_list


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=50, power=0.9):
    """
    Polynomial decay of learning rate
    :param init_lr is base learning rate
    :param iter is a current iteration
    :param lr_decay_iter how frequently decay occurs, default is 1
    :param max_iter is number of maximum iterations
    :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          epochs: int, lr: float, train_std: float, train_mean: float, val_std: float, val_mean: float,
          log: logging, criterion=None):
    """
    Train model on train_loader and track the validation value on val_loader.

    :param model: training neural network model
    :param train_loader: train set loader
    :param val_loader: validation set loader
    :param epochs: training epoches
    :param lr: learning rate
    :param train_std: training set's standard deviation
    :param train_mean: training set's average value
    :param val_std: validation set's standard deviation
    :param val_mean: validation set's average value
    :param log: logger object
    :param criterion: loss function (default: L1 loss)
    :return: loss_list, rmse_list, val_loss_list, val_rmse_list
    """
    with open('log_train.txt', 'w') as log_tf, open('log_val.txt', 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')
    
    # loss function
    criterion = nn.NLLLoss() if criterion is None else criterion
    print('criterion',criterion)
    criterion = criterion
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # track loss list
    loss_list, acc_list, val_loss_list, val_acc_list = tuple([] for _ in range(4))
    total_steps = len(train_loader)
    
    for epoch in tqdm(range(epochs)):
        # start training
        model.train()
        poly_lr_scheduler(optimizer, lr, epoch+1)
        counter = 0
        for ix, (x, y, x_vol, x_mkt, x_geo, x_crop) in enumerate(train_loader):
            counter+=1
            # track starting time
            start_time = time.time()

            # clear the accumulated gradient before each instance
            model.zero_grad()

            x, y, x_vol, x_geo, x_mkt, x_crop = x.float(), y.float(), x_vol.float(), x_geo.float(), x_mkt.long(), x_crop.long()
            # prepare the data and label
            x, y, x_vol, x_geo, x_mkt, x_crop = x.to(device), y.to(device), x_vol.to(device),\
                                                x_geo.to(device), x_mkt.to(device), x_crop.to(device)
            
            
#             print('x.shape',x.shape)
#             print('x_vol.shape',x_vol.shape)
#             print('x_geo.shape',x_geo.shape)
#             print('x_mkt.shape',x_mkt.shape)
#             print('x_crop.shape',x_crop.shape)
            # run the forward pass
            outputs = model(x, x_vol, x_geo, x_mkt, x_crop)

            # Compute the loss, gradients, and update the parameters
            #outputs = outputs.view(-1)
            y = y.view(-1).long()
            loss = criterion(outputs, y)
            # back propogation
            loss.backward()
            # update parameters
            optimizer.step()
            # clear gradient
            optimizer.zero_grad()
            
            acc = cal_acc(outputs, y)
            acc_list.append(acc)
            loss_list.append(loss)
            # append loss to the loss list
            #rmse = RMSE(outputs, y.float(), train_std, train_mean)
            #rmse_list.append(rmse)
            

            # print in every 50 episodes
            if (ix+1) % 50 == 0:
                log.info(f'Epoch [{epoch + 1}/{epochs}], Step [{ix + 1}/{total_steps}], '
                      f'Time [{time.time() - start_time:.6f} sec], Avg loss: {sum(loss_list[-50:])/50: .4f}, '
                      f'Avg acc: {sum(acc_list[-50:])/50: .4f}')
        print('-(counter)',-(counter))
        avg_loss = sum(loss_list[-(counter):])/counter
        avg_acc = sum(acc_list[-(counter):])/counter
        loss_list.append(avg_loss)
        acc_list.append(avg_acc)
        log.info(f"train acc: {avg_acc: .4f}; avg loss: {avg_loss: .4f}")
        # validation
        model.eval()
        log.info("Validating on the testing set...")
        # TODO: validate crops separately
        val_acc, val_loss = validate(model, val_loader)
        val_avg_loss = np.mean(val_loss)
        val_avg_acc = np.mean(val_acc)
        val_loss_list.append(val_avg_loss)
        val_acc_list.append(val_avg_acc)
        log.info(f"validation acc: {val_avg_acc: .4f}; avg loss: {val_avg_loss: .4f}")
        
        with open('log_train.txt', 'a') as log_tf, open('log_val.txt', 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch, loss=avg_loss, accu=100*avg_acc))
            log_vf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch, loss=val_avg_loss, accu=100*val_avg_acc))

#     return loss_list, rmse_list, val_loss_list, val_rmse_list
    return loss_list, acc_list, val_loss_list, val_acc_list
