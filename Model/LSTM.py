import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, seq_len=90, output_len=30,
                 layerNum=3, drop_out=0., bidirection=False):

        super(LSTM, self).__init__()
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
        # bidirection
        self.bidirection = bidirection

        # fc embedding
        # self.fc_embedding = nn.Linear(self.inputDim, self.inputDim)
        self.fc_embedding = nn.Conv1d(4, 4, 1)
        self.batch_norm = nn.BatchNorm1d(4)

        # LSTM cell
        self.lstm = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                            num_layers=self.layerNum, dropout=drop_out,
                            batch_first=True, bidirectional=bidirection)
        self.dropout = nn.Dropout(drop_out)
        self.layer_norm = nn.LayerNorm(hiddenNum * (bidirection + 1))

        self.fc = nn.Linear(hiddenNum * (bidirection + 1), self.outputDim)
        self.final_fc = nn.Linear(self.seq_len, self.output_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.batch_norm(self.fc_embedding(x))
        x = x.transpose(2, 1)

        # x = [batch, input_len, output_len]
        batch_size = x.size(0)
        h0 = torch.zeros(self.layerNum * (self.bidirection + 1), batch_size, self.hiddenNum)
        c0 = torch.zeros(self.layerNum * (self.bidirection + 1), batch_size, self.hiddenNum)
        h0, c0 = h0.to(device), c0.to(device)

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

        # fc_output = self.fc(output[:, -1, :])

        return fc_output

    def weight_init(self):
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


#%%

model = LSTM(inputDim=4, hiddenNum=64, outputDim=1, layerNum=3, bidirection=True)
model.weight_init()
model.to(device)

x = torch.rand(size=(128, 90, 4))
output = model(x)
print(output.size())
