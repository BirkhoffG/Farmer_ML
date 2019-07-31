import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, inputDim, hiddenNum, outputDim, seq_len=90, output_len=30, layerNum=3, drop_out=0.):

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

        # LSTM cell
        self.lstm = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                            num_layers=self.layerNum, batch_first=True)
        self.atten = ScaledDotProductAttention(dropout=drop_out)
        self.dropout = nn.Dropout(drop_out)
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

        # fc_output = F.relu(self.fc(atten_output))
        fc_output = self.fc(atten_output)
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


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


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


#%%
hidden_num = 8

model = LSTM(inputDim=1, hiddenNum=hidden_num, outputDim=1, layerNum=3)
model.weight_init()
model.to(device)

x = torch.rand(size=(64, 90, 1))
output = model(x)
print(output.size())
