import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import time
import numpy as np
import logging
from Train.utils import device, list2arr, val_RMSE, RMSE


def validate(model: nn.Module, val_loader: DataLoader, std, mean):
    """
    validate the test set in the model
    :param model: neural network modal
    :param val_loader: test loader
    :param std: standard deviation
    :param mean: mean value
    :return: RMSE on validation set; loss list on validation set
    """
    # track values
    pred_list, tar_list, loss_list = [], [], []
    # L2 loss
    criterion = nn.MSELoss()
    # model evaluation
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            result = model(x.float())

            result_list = result.cpu().data.numpy().tolist()
            pred_list.extend(result_list)
            tar_list.extend(y.cpu().data.numpy().tolist())

            loss = criterion(result, y.float()).cpu().data.numpy()
            loss_list.append(loss)

    pred_arr = list2arr(pred_list)
    tar_arr = list2arr(tar_list)
    assert pred_arr.shape == tar_arr.shape

    rmse = val_RMSE(pred_arr, tar_arr, std, mean)
    return rmse, loss_list


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
    # loss function
    criterion = nn.L1Loss() if criterion is None else criterion
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # track loss list
    loss_list, rmse_list, val_loss_list, val_rmse_list = tuple([] for _ in range(4))
    total_steps = len(train_loader)

    for epoch in range(epochs):
        # start training
        model.train()
        poly_lr_scheduler(optimizer, lr, epoch+1)

        for ix, (x, y) in enumerate(train_loader):
            # track starting time
            start_time = time.time()

            # clear the accumulated gradient before each instance
            model.zero_grad()

            # prepare the data and label
            x, y = x.to(device), y.to(device)

            # run the forward pass
            outputs = model(x.float())

            # Compute the loss, gradients, and update the parameters
            loss = criterion(outputs, y.float())

            # back propogation
            loss.backward()
            # update parameters
            optimizer.step()
            # clear gradient
            optimizer.zero_grad()

            # append loss to the loss list
            rmse = RMSE(outputs, y.float(), train_std, train_mean)
            rmse_list.append(rmse)
            loss_list.append(loss)

            # print in every 50 episodes
            if (ix+1) % 50 == 0:
                log.info(f'Epoch [{epoch + 1}/{epochs}], Step [{ix + 1}/{total_steps}], '
                      f'Time [{time.time() - start_time} sec], Avg loss: {sum(loss_list[-50:])/50}, '
                      f'Avg RMSE: {sum(rmse_list[-50:])/50}')

        # validation
        model.eval()
        log.info("Validating on the testing set...")
        val_rmse, val_loss_list = validate(model, val_loader, val_std, val_mean)
        val_avg_loss = np.mean(val_loss_list)
        val_rmse_list.append(val_rmse)
        val_loss_list.append(val_avg_loss)
        log.info(f"RMSE: {val_rmse}; avg loss: {val_avg_loss}")

    return loss_list, rmse_list, val_loss_list, val_rmse_list
