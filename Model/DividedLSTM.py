import torch.nn as nn
import torch
import torch.nn.functional as F
from Model.TCN import TCN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DividedLSTM(nn.Module):

    def __init__(self, input_dim, hidden_num, output_dim, seq_len=90, output_len=1,
                 layer_num=1, drop_out=0., bidirection=False):
        super(DividedLSTM, self).__init__()
        self.seq_len = seq_len
        self.layer_num = layer_num
        self.bidirection = bidirection
        self.hidden_num = hidden_num

        self.lstm_bottom = nn.LSTM(input_size=input_dim, hidden_size=hidden_num,
                                   num_layers=layer_num, dropout=drop_out,
                                   batch_first=True)
        self.tcn = TCN(input_size=hidden_num, output_size=1,
                       num_channels=[hidden_num, hidden_num // 4, output_dim],
                       input_len=90, output_len=1)

        self.dropout = nn.Dropout(drop_out)
        self.layer_norm = nn.LayerNorm(hidden_num)

        self.fc = nn.Linear(hidden_num // 4, output_dim)
        self.final_fc = nn.Linear(seq_len, output_len)

    def forward(self, x):
        # x = [batch, input_len, output_len]
        batch_size = x.size(0)
        h0 = torch.zeros(self.layer_num, batch_size, self.hidden_num)
        c0 = torch.zeros(self.layer_num, batch_size, self.hidden_num)
        h0, c0 = h0.to(device), c0.to(device)
        # output = [batch_size, seq_len, hidden_num]
        # hn = ([num_layer, batch_len, hidden_num],
        #       [num_layer, batch_len, hidden_num])
        x = self.layer_norm(self.lstm_bottom(x, (h0, c0))[0])
        print(f"x.size: {x.size()}")

        y = self.tcn(x)
        return y

        # fc_output = self.fc(y)
        # fc_output = self.dropout(fc_output)
        # fc_output = fc_output.squeeze()
        # fc_output = self.final_fc(fc_output)
        #
        # return fc_output

    def weight_init(self):
        for lstm in [self.lstm_bottom]:
            for param in lstm.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)


model = DividedLSTM(input_dim=4, hidden_num=32, output_dim=1)
model.weight_init()
model.to(device)

x = torch.rand(size=(128, 90, 4))
output = model(x)
print(output.size())
