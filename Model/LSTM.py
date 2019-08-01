import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len=90, output_len=30, layer_num=3, heads=4, drop_out=0.):

        super(LSTM, self).__init__()
        # hidden cell numbers
        self.hiddenNum = hidden_dim
        # input dimension
        self.inputDim = input_dim
        # output dimension
        self.outputDim = output_dim
        # layer number
        self.layerNum = layer_num
        # sequence length
        self.seq_len = seq_len
        # output length
        self.output_len = output_len

        # LSTM cell
        self.lstm = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                            num_layers=self.layerNum, batch_first=True)
        self.atten = ScaledDotProductAttention(dropout=drop_out)
        # self.atten = MultiHeadAttention(model_dim=hidden_dim, num_heads=heads, dropout=drop_out)
        self.dropout = nn.Dropout(drop_out)
        self.feedforward = PositionalWiseFeedForward(hidden_dim, hidden_dim*4, drop_out)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)
        self.final_fc = nn.Linear(self.seq_len, self.output_len)

    def forward(self, x):
        # x = [batch, input_len, output_len]
        batch_size = x.size(0)
        h0 = torch.zeros(self.layerNum * 1, batch_size, self.hiddenNum)
        c0 = torch.zeros(self.layerNum * 1, batch_size, self.hiddenNum)
        h0, c0 = h0.to(device), c0.to(device)

        # output = [batch_size, seq_len, hidden_num]
        # hn = ([num_layer, batch_len, hidden_num],
        #       [num_layer, batch_len, hidden_num])
        output, hn = self.lstm(x, (h0, c0))

        atten_output, _ = self.atten(output, output, output)
        # print(f"atten_output size: {atten_output.size()}")

        # print(f"atten_output: {atten_output.size()}")
        position_wised_output = self.feedforward(atten_output)
        fc_output = self.fc(position_wised_output)
        # print(f"fc_output: {fc_output.size()}")
        fc_output = self.dropout(fc_output)
        fc_output = fc_output.squeeze()
        fc_output = self.final_fc(fc_output)

        # fc_output = self.fc(output[:, -1, :])

        return fc_output

    def weight_init(self):
        # TODO
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v,):
        attention = torch.bmm(q, k.transpose(1, 2))

        scale = k.size(-1) ** -0.5
        attention = attention * scale
        # calculate softmax
        attention = self.softmax(attention)
        # add dropout
        attention = self.dropout(attention)
        # context
        context = torch.bmm(attention, v)
        return context, attention


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        print(f"before transpose: {x.size()}")
        output = x.transpose(1, 2)
        print(f"after transpose: {output.size()}")
        print(f"conv1: {self.w1(output).size()}")
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))
        print(f"output size: {output.size()}")

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

#%%
hidden_num = 512

model = LSTM(input_dim=1, hidden_dim=128, output_dim=1, layer_num=3)
model.weight_init()
model.to(device)

x = torch.rand(size=(128, 90, 1))
output = model(x)
print(output.size())
