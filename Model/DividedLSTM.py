import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DividedLSTM(nn.Module):

    def __init__(self, input_dim, hidden_num, output_dim, seq_len=90, output_len=1,
                 layer_num=3, drop_out=0., bidirection=True):
        super(DividedLSTM, self).__init__()
        self.seq_len = seq_len

        self.lstm_bottom = nn.LSTM(input_size=input_dim, hidden_size=hidden_num,
                                   num_layers=layer_num, dropout=drop_out,
                                   batch_first=True)

        self.lstm_top = nn.LSTM(input_size=hidden_num, hidden_size=hidden_num,
                                num_layers=layer_num, dropout=drop_out,
                                batch_first=False)

        self.dropout = nn.Dropout(drop_out)
        self.layer_norm = nn.LayerNorm(hidden_num * (bidirection + 1))

        self.fc = nn.Linear(hidden_num * (bidirection + 1), output_dim)
        self.final_fc = nn.Linear(seq_len, output_len)

    def forward(self, x):
        # truncate input
        scale = self.seq_len // 3
        x_0, x_1, x_2 = x[:scale], x[scale: 2*scale], x[2*scale:]

        # x = [batch, input_len, output_len]
        batch_size = x.size(0)
        h0 = torch.zeros(self.layerNum * (self.bidirection + 1), batch_size, self.hiddenNum)
        c0 = torch.zeros(self.layerNum * (self.bidirection + 1), batch_size, self.hiddenNum)
        h0, c0 = h0.to(device), c0.to(device)

        # truncated output
        output_0, _ = self.lstm_bottom(x_0, (h0, c0))
        output_1, _ = self.lstm_bottom(x_1, (h0, c0))
        output_2, _ = self.lstm_bottom(x_2, (h0, c0))

        output = torch.cat((output_0, output_1, output_2))

        # output = [batch_size, seq_len, hidden_num]
        # hn = ([num_layer, batch_len, hidden_num],
        #       [num_layer, batch_len, hidden_num])
        output, hn = self.lstm(x, (h0, c0))
        output = self.layer_norm(output)

        #         output = output.view(output.size(0)*output.size(1), output.size(2))
        fc_output = self.fc(output)
        fc_output = self.dropout(fc_output)
        fc_output = fc_output.squeeze()
        fc_output = self.final_fc(fc_output)

        return fc_output