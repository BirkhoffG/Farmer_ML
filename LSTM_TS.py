from src.util import load_data, createSamples, divideTrainTest, create_multi_ahead_samples
from src.ts_loader import Time_Series_Data

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader


import time
import numpy as np
import pandas as pd

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

ts, data = load_data('./dataset/brinjal_price.csv', columnName='Howly_Assa')


#%%

crop_df = pd.read_csv('./dataset/brinjal_price.csv', index_col=0, parse_dates=True, low_memory=False)


def data_preprocess(data, lag, h_train, h_test):
    trainData, testData = divideTrainTest(data)
    rnn_format = True

    trainX, trainY = create_multi_ahead_samples(trainData, lag, h_train, RNN=rnn_format)
    testX, testY = create_multi_ahead_samples(testData, lag, h_test, RNN=rnn_format)

    trainY = np.squeeze(trainY).reshape(-1, h_train)
    testY = np.squeeze(testY).reshape(-1, h_test)
    return trainX, trainY, testX, testY


train_x_arr, train_y_arr, test_x_arr, test_y_arr = [], [], [], []

for col in crop_df.columns:
    # ts, data = load_data('./dataset/brinjal_price.csv', columnName=col)
    print(f"col: {col}")
    data = crop_df[col].values.reshape(-1, 1).astype("float32")
    train_x, train_y, test_x, test_y = data_preprocess(data, lag=60, h_train=1, h_test=1)
    train_x_arr.append(train_x)
    train_y_arr.append(train_y)
    test_x_arr.append(test_x)
    test_y_arr.append(test_y_arr)

#%%

print(f"converting train_x...")
train_x = np.array(train_x_arr)
print(f"train_x shape: {train_x.shape}")

print(f"converting train_y...")
train_y = np.array(train_y_arr)
print(f"train_y shape: {train_y.shape}")

print(f"converting test_x...")
test_x = np.array(test_x_arr)
print(f"test_x shape: {test_x.shape}")

print("converting test_y...")
test_y = np.array(test_y_arr)
print(f"test_y shape: {test_y.shape}")

#%%

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data)

trainData, testData = divideTrainTest(dataset)

rnn_format = True

trainX, trainY = create_multi_ahead_samples(trainData, 60, 30, RNN=rnn_format)
testX, testY = create_multi_ahead_samples(testData, 60, 30, RNN=rnn_format)

trainY = np.squeeze(trainY).reshape(-1, 30)
testY = np.squeeze(testY).reshape(-1, 30)
print("train X shape:", trainX.shape)
print("train y shape:", trainY.shape)
print("test X shape:", testX.shape)
print("test y shape:", testY.shape)

batch_size = 32
# train_dataset = Time_Series_Data(trainX, trainY)
train_dataset = Time_Series_Data(trainX, trainY)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None,
                                           batch_sampler=None, num_workers=1)

test_dataset = Time_Series_Data(testX, testY)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, sampler=None,
                                          batch_sampler=None, num_workers=1)
print(f"train loader len: {len(train_loader)}")
print(f"test loader len: {len(test_loader)}")

#%%

# load data

train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')

#%%

train_x_shape = train_x.shape
train_x = train_x.reshape(train_x_shape[0], train_x_shape[1], 1)

batch_size = 128

train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                          batch_size=batch_size, shuffle=True, num_workers=1)

print(f"train loader len: {len(train_loader)}")

#%%

class LSTM(nn.Module):
    def __init__(self, inputDim, hiddenNum, outputDim, layerNum=3, drop_out=0.):

        super(LSTM, self).__init__()
        # hidden cell numbers
        self.hiddenNum = hiddenNum
        # input dimension
        self.inputDim = inputDim
        # output dimension
        self.outputDim = outputDim
        # layer number
        self.layerNum = layerNum
        # LSTM cell
        self.lstm = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                            num_layers=self.layerNum, dropout=drop_out,
                            batch_first=True)

        print(self.lstm)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)

    def forward(self, x):
        batch_size = x.size(0)
        print(x.shape)
        h0 = torch.zeros(self.layerNum * 1, batch_size, self.hiddenNum)
        print(h0.dtype)
        c0 = torch.zeros(self.layerNum * 1, batch_size, self.hiddenNum)
        print(c0.dtype)
        #         if self.cuda:
        #             h0.cuda()
        #             c0.cuda()
        # lstm cell
        output, hn = self.lstm(x, (h0, c0))
        hn = hn[0].view(batch_size, self.hiddenNum)
        fc_output = self.fc(hn)
        return fc_output



#%%

hidden_num = 365

model = LSTM(inputDim=1, hiddenNum=hidden_num, outputDim=1)
# model.to(device)

#%%

learning_rate = 0.01
epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#%%

loss_list = []
total_steps = len(train_loader)

model.train()
for epoch in range(epochs):
    for i, (x, y) in enumerate(train_loader):
        epoch_start_time = time.time()

        # TODO.md train on gpu
        # train on gpu
        x, y = x.to(device), y.to(device)

        # run the forward pass
        outputs = model(x.float())
        print(f"outputs dtype: {outputs.dtype}")
        print(f"target dtype: {y.dtype}")

        loss = criterion(outputs, y.float())
        loss_list.append(loss)

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{total_steps}], '
                  f'Time [{time.time() - epoch_start_time} sec], Loss: {loss.item()}')
