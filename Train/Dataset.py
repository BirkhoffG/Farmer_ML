from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class MarketDataset(Dataset):

    def __init__(self, p_arr: np.array, v_arr: np.array, w_arr: np.array, geo_loc: np.array,
                 pro_pop: np.array, one_hot=True):
        super(MarketDataset, self).__init__()
        # price tensor
        self.p_tensor = torch.from_numpy(p_arr)
        # volume tensor
        self.v_tensor = torch.from_numpy(v_arr)
        # weather tensor
        self.w_tensor = torch.from_numpy(w_arr)
        # geometric location
        self.geo_loc = torch.from_numpy(geo_loc)
        # production and population
        self.pro_pop = torch.from_numpy(pro_pop)
        # one hot encoding
        self.one_hot = one_hot

    def __getitem__(self, index):
        p = self.p_tensor[index]
        # TODO
        p = torch.cat((p, torch.eye(366)))
        v = self.v_tensor[index]
        w = self.w_tensor[index]
