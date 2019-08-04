import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionLSTM(nn.Module):
    def __init__(self, inputDim, hiddenNum, outputDim, seq_len=90, output_len=30, layerNum=3, drop_out=0.):

        super(AttentionLSTM, self).__init__()
        # hidden cell numbers
        self.hiddenNum = hiddenNum
        # input dimension
        self.inputDim = inputDim
        # output dimension
        self.outputDim = outputDim
        # layer number
        self.layerNum = layerNum
        # sequence length
        self.seq_len = seq_len
        # output length
        self.output_len = output_len

        # LSTM cell
        self.lstm = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                            num_layers=self.layerNum, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.hiddenNum)
        self.atten = ScaledDotProductAttention(dropout=drop_out)
        # self.dropout = nn.Dropout(drop_out)
        # self.fc = nn.Linear(self.hiddenNum, self.outputDim)
        self.lstm_decoder = nn.LSTM(input_size=self.hiddenNum, hidden_size=self.hiddenNum,
                                    num_layers=self.layerNum, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(self.hiddenNum, self.hiddenNum),
            nn.Tanh(),
            nn.Linear(self.hiddenNum, self.outputDim),
            nn.Dropout(drop_out),
            nn.LayerNorm(self.outputDim),
        )
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
        output = self.layer_norm(output)

        atten_output, _ = self.atten(output, output, output)
        # print(f"atten_output size: {atten_output.size()}")

        decoder_output, hn_d = self.lstm_decoder(atten_output, hn)
        decoder_output = self.layer_norm(decoder_output)
        fc_output = self.dense(decoder_output)

        # fc_output = self.fc(decoder_output)
        # fc_output = self.dropout(fc_output)
        fc_output = fc_output.squeeze()
        fc_output = self.final_fc(fc_output)

        # fc_output = self.fc(output[:, -1, :])

        return fc_output

    def weight_init(self):
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

    def forward(self, q, k, v, scale=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
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
        output = x.transpose(1, 2)
        output = self.w2(F.tanh(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output



#%%
hidden_num = 8

model = AttentionLSTM(inputDim=1, hiddenNum=hidden_num, outputDim=1, layerNum=3)
model.weight_init()
model.to(device)

x = torch.rand(size=(64, 90, 1))
output = model(x)
print(output.size())
