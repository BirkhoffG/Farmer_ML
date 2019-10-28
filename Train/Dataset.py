from torch.utils.data.dataset import TensorDataset, Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import logging


class ArrayDataset(TensorDataset):

    def __init__(self, *arrs):
        super(ArrayDataset, self).__init__()
        # init tensors
        self.tensors = [torch.from_numpy(arr) for arr in arrs]
        assert all(self.tensors[0].size(0) == tensor.size(0) for tensor in self.tensors)

    def data_loader(self, batch_size=128, shuffle=False, num_workers=4):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class DirDataset(ArrayDataset):

    def __init__(self, *path):
        # TODO read np arr from path
        arrs = [].extend(lambda x: self.load_arrays(_path) for _path in path)
        super(DirDataset, self).__init__(arrs)

    @staticmethod
    def load_arrays(path):
        try:
            arr_list = os.listdir(path)
            return tuple(np.load(f'{path}/{arr}') for arr in arr_list if ('.npy' in arr) or ('.npz' in arr))
        except (FileExistsError, FileNotFoundError, NotADirectoryError) as e:
            logging.error(e)
            return None


class MarketDataset(ArrayDataset):

    def __init__(self, p_arr: np.array, v_arr: np.array, w_arr: np.array, geo_loc: np.array,
                 pro_pop: np.array, y_arr: np.array, one_hot=True):
        super(MarketDataset, self).__init__(p_arr, v_arr, w_arr, geo_loc, pro_pop, y_arr)
        # one hot encoding
        self.one_hot = one_hot

    def __getitem__(self, index):
        p, v, w, g, pp, y = tuple(tensor[index] for tensor in self.tensors)

        if self.one_hot:
            p = torch.cat((p, torch.eye(366)))
            v = torch.cat((v, torch.eye(366)))
            w = torch.cat((w, torch.eye(366)))
        return p, v, w, g, pp, y

