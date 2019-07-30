import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers=n_layers,
                          dropout=dropout, bidirectional=False)

    def forward(self, input, hidden=None):
        outputs, hidden = self.gru(input, hidden)
        return outputs, hidden

    def init_hidden(self, input_len):
        return torch.zeros(1*self.n_layers, input_len, self.hidden_size, device=device)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / np.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        h = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + output_size, hidden_size, n_layers, dropout)
        self.fc_out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # input: (1, B, V)
        # calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        # (B, 1, N)
        context = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1))
        # (1, B, N)
        context = context.transpose(0, 1)
        # combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((input, context), dim=2)

        output, hidden = self.gru(rnn_input, last_hidden)
        # (1, B, V) -> (B, V)
        output = output.squeeze(0)
        return output, hidden


#%%
# x = (seq_len, batch, input_dim)
x = torch.rand(size=(90, 64, 2))

encoder = Encoder(input_size=2, hidden_size=8, n_layers=3)
hidden = encoder.init_hidden(x.size(1))
# output = (seq_len, batch, input_dim)
# hidden = (layer_n, batch, input_dim)
output, hidden = encoder(x, hidden)

print(f"output size: {output.size()}")
print(f"hidden size: {hidden.size()}")

decoder = Decoder(hidden_size=8, output_size=1, n_layers=3)
decoder_output, _ = decoder(output, last_hidden=hidden, encoder_outputs=output)
print(decoder_output)

