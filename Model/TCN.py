import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from torch.nn.utils import weight_norm
from Model.AttenModel import ScaledDotProductAttention as Attention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, input_len, output_len,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, tied_weights=False):
        super(TCN, self).__init__()

        # TCN nets
        self.p_tcn = TemporalConvNet(1, num_channels, kernel_size, dropout=dropout)
        self.p_attention = Attention(dropout)
        self.v_tcn = TemporalConvNet(1, num_channels, kernel_size, dropout=dropout)
        self.v_attention = Attention(dropout)

        # 定义最后线性变换的纬度，即最后一个卷积层的通道数（类似2D卷积中的特征图数）到所有词汇的映射
        self.decoder = nn.Linear(num_channels[-1], output_size)
        self.final_fc = nn.Sequential(
            nn.Linear(input_len * 2, input_len * 2),
            nn.Dropout(dropout),
            nn.Linear(input_len * 2, output_len),
        )

        # 对输入词嵌入执行Dropout 表示随机从句子中舍弃词，迫使模型不依赖于单个词完成任务
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        # self.encoder.weight.data.normal_(0, 0.01)
        nn.init.xavier_normal_(self.decoder.weight.data)
        nn.init.normal_(self.decoder.bias.data)
        # self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        # emb = self.drop(input)
        price_x, volume_x = x[:, :, :1], x[:, :, 1:]
        price_y = self.p_tcn(price_x.transpose(1, 2)).transpose(1, 2)
        price_y, _ = self.p_attention(price_y, price_y, price_y)
        volume_y = self.v_tcn(volume_x.transpose(1, 2)).transpose(1, 2)
        volume_y, _ = self.v_attention(volume_y, volume_y, volume_y)
        # print(f"price_y shape: {price_y.size()}; volume_y shape: {volume_y.size()}")
        y = torch.cat((price_y, volume_y), dim=1)
        # print(f"y shape: {y.size()}")
        # y = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        y = self.final_fc(y[:, :, -1])
        # print("y.size: ", y.size())
        return y


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # init conv1
        nn.init.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2))
        # nn.init.normal_(self.conv1.weight.data)
        if self.conv1.bias is not None:
            nn.init.normal_(self.conv1.bias.data)
        # init conv2
        nn.init.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2))
        # nn.init.normal_(self.conv2.weight.data)
        if self.conv2.bias is not None:
            nn.init.normal_(self.conv2.bias.data)
        if self.downsample is not None:
            nn.init.xavier_uniform_(self.downsample.weight, gain=np.sqrt(2))
            # nn.init.normal_(self.downsample.weight.data)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


#%%

model = TCN(input_size=2, output_size=1, num_channels=[16, 8, 4], input_len=90, output_len=1)
# model.weight_init()
model.to(device)

x = torch.rand(size=(128, 90, 2))
output = model(x)
print(output.size())
