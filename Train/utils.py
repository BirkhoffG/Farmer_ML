import torch
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import logging

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def RMSE(output: torch.tensor, target: torch.tensor, std: float, mean: float):
    """
    $$ RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(\hat{y}_t^2-y_t^2)}$$

    :param output: model predicted tensor
    :param target: target tensor
    :param std: standard deviation
    :param mean: mean
    :return:
    """
    normalized_output = output * std + mean
    normalized_target = target * std + mean

    return torch.sqrt(torch.mean((normalized_output - normalized_target) ** 2))


def val_RMSE(pred_arr, tar_arr, std, mean):
    nomalized_pre_arr = pred_arr * std + mean
    nomalized_test_y = tar_arr * std + mean
    #     print(f"predict array shape: {nomalized_pre_arr.shape}; target array shape: {nomalized_test_y.shape}")
    return np.sqrt(mse(nomalized_pre_arr, nomalized_test_y))


def list2arr(li: list):
    """
    convert list to numpy array
    :param li: list
    :return: converted np array
    """
    arr = np.array(li)
    return arr.reshape(arr.shape[0], arr.shape[1])


def norm(arr, log: logging, std=None, mean=None):
    std = np.std(arr) if std is None else std
    mean = np.mean(arr) if mean is None else mean
    log.info(f"std = {std}, mean = {mean}")
    return (arr - mean) / std, std, mean


def norm_arrays(*arrays, log: logging, std, mean):
    log.info(f"std = {std}, mean = {mean}")
    return tuple((arr - mean) / std for arr in arrays)


def stat(arr):
    print(f"std: {np.std(arr)}; mean: {np.mean(arr)}")


def plot(li, metric: str):
    print("Val value: ", min(li))
    plt.plot(range(1, len(li) + 1), li)
    plt.ylabel(metric)
    plt.title(f'{metric} plot')
