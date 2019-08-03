import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SelfAttentionModel(nn.Module):

    def __init__(self, input_len, input_dim, output_len, output_dim, layer_num=3, cycle=None, dropout=0.):
        super(SelfAttentionModel, self).__init__()

        if cycle is None:
            cycle = input_len

        self.positionalEncode = PositionalEncoding(hidden_num=input_dim, input_len=input_len, cycle=cycle)
        self.attention_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=input_dim, dropout=dropout) for _ in range(layer_num)]
        )

        self.feedforward = FeedForward(input_dim, input_len, output_dim, output_len, dropout)

    def forward(self, x):
        output = self.positionalEncode(x)
        attentions = []

        for layer in self.attention_layers:
            output, attention = layer(output)
            attentions.append(attention)

        output = self.feedforward(output)
        return output, attentions


class FeedForward(nn.Module):

    def __init__(self, input_dim, input_len, output_dim, output_len, dropout=0.):
        super(FeedForward, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.final = nn.Linear(input_len, output_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.dropout(self.fc(x))
        output = output.squeeze()
        return self.dropout(self.final(output))


class AttentionLayer(nn.Module):

    def __init__(self, hidden_dim, dropout=0.):
        super(AttentionLayer, self).__init__()

        self.atten = ScaleDotProductAttention(hidden_dim=hidden_dim, dropout=dropout)
        self.feedforward = PositionWiseFeedForward(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x):
        # self attention
        context, attention = self.atten(x, x, x)
        # feed forward network
        output = self.feedforward(context)

        return output, attention


class ScaleDotProductAttention(nn.Module):

    def __init__(self, hidden_dim, dropout=0.):
        super(ScaleDotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        # attention equation
        attention = torch.bmm(q, k.transpose(1, 2))
        # scale term
        scale = np.sqrt(1/self.hidden_dim)
        attention = attention * scale
        # compute soft max
        attention = self.softmax(attention)
        # add dropout
        attention = self.dropout(attention)
        # context = attention * v
        context = torch.bmm(attention, v)
        return context, attention


class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, input_dim, hidden_dim=None, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 4
        self.w_1 = nn.Linear(input_dim, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        # employ residual connection
        return self.layer_norm(x + output)


class PositionalEncoding(nn.Module):

    def __init__(self, hidden_num, input_len, cycle=None):
        super(PositionalEncoding, self).__init__()

        if cycle is None:
            self.cycle = input_len

        position_encoding = torch.tensor([
            [2 * t * np.pi / cycle for _ in range(hidden_num)]
            for t in range(input_len)
        ], requires_grad=False)

        # even dim -> sin; odd dim -> cos
        position_encoding[:, 0::2] = torch.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = torch.cos(position_encoding[:, 1::2])

        self.register_buffer('position_encoding', position_encoding)

    def forward(self, x):
        x = x + self.position_encoding[:, :x.size(1)]
        return x

#%%


if __name__ == '__main__':
    x = torch.rand(size=(64, 90, 4))
    # pe = PositionalEncoding(hidden_num=4, input_len=90, cycle=90)
    atten = SelfAttentionModel(input_len=90, input_dim=4,
                               output_len=1, output_dim=1, layer_num=3)
    output = atten(x)
    print(output[0].size())

