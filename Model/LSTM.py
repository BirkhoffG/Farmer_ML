import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, inputDim, hiddenNum, outputDim, seq_len=365, layerNum=3, drop_out=0.):

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

        #         print(self.lstm)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.layerNum * 1, batch_size, self.hiddenNum)
        c0 = torch.zeros(self.layerNum * 1, batch_size, self.hiddenNum)
        h0, c0 = h0.to(device), c0.to(device)

        # output = [batch_size, seq_len, hidden_num]
        # hn = ([num_layer, batch_len, hidden_num],
        #       [num_layer, batch_len, hidden_num])
        output, hn = self.lstm(x, (h0, c0))

        #         output = output.view(output.size(0)*output.size(1), output.size(2))
        fc_output = self.fc(output[:, -1, :])

        return fc_output

    def weight_init(self):
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


hidden_num = 8

model = LSTM(inputDim=1, hiddenNum=hidden_num, outputDim=1, layerNum=3)
model.weight_init()
model.to(device)