import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_num, output_dim, seq_len=90, output_len=30, num_layers=3, drop_out=0.):
        super(GRU, self).__init__()

        self.num_layers = num_layers
        self.hidden_num = hidden_num

        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_num,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_num, output_dim)
        self.final_fc = nn.Linear(seq_len, output_len)

    def forward(self, inputs):
        # inputs = [batch, input_len, features]
        batch_size = inputs.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_num).to(device)
        # h0.to(device)

        output, hn = self.gru(inputs, h0)

        output = self.fc(output)
        return self.final_fc(output.squeeze())

    def weight_init(self):
        for param in self.gru.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


#%%

model = GRU(input_dim=4, hidden_num=64, output_dim=1, output_len=1)
model.weight_init()
model.to(device)

x = torch.rand(size=(64, 90, 4))
output = model(x)
print(output.size())