import torch.nn as nn
import torch
from Model.TCN import TemporalConvNet


class WideDeepTCN(nn.Module):

    def __init__(self, input_len, output_len,
                 mkt_num, mkt_emb,
                 crop_num=6, crop_emb=3,
                 num_channels=None, kernel_size=3, dropout=0.3,
                 p_tcn=None, v_tcn=None, ):
        '''

        :param num_channels: number of channels in TCN block
        :param input_len: input sequence length
        :param output_len: output sequence length
        :param mkt_num: market size
        :param mkt_emb: embedded market size
        :param crop_num: number of crops
        :param crop_emb: embedded crop size
        :param kernel_size: kernel size in TCN block
        :param dropout:
        :param p_tcn: TCN for price seq
        :param v_tcn: TCN for volume seq
        '''

        super(WideDeepTCN, self).__init__()

        if num_channels is None:
            num_channels = [16, 8, 4, 1]

        # TCN nets
        self.p_tcn = TemporalConvNet(1, num_channels, kernel_size, dropout=dropout) if p_tcn is None else p_tcn
        self.v_tcn = TemporalConvNet(1, num_channels, kernel_size, dropout=dropout) if v_tcn is None else v_tcn

        # market embedding
        self.mkt_embedding = nn.Embedding(mkt_num, mkt_emb)

        # crop embedding: 6 crops, 3 dimension
        self.crop_embedding = nn.Embedding(crop_num, crop_emb)

        self.wide_fc = nn.Linear(2 * input_len, 2 * input_len)
        # TODO: not deep enough
        self.deep_fc = nn.Sequential(
            nn.Linear(2 * input_len + mkt_emb + crop_emb + 2, input_len + mkt_emb + crop_emb + 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_len + mkt_emb + crop_emb + 2, input_len // 2 + mkt_emb // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_len // 2 + mkt_emb // 2, input_len // 3 + mkt_emb // 3),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.final_fc = nn.Linear(input_len * 2 + input_len // 3 + mkt_emb // 3, output_len)
        self.log_softmax = nn.LogSoftmax(dim=1)
        # TODO: adjust initialization method
        # self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.final_fc.weight.data)
        nn.init.normal_(self.final_fc.bias.data)

    def forward(self, price_x, volume_x, geo_x, mkt_x, crop_x):
        # TCN
#         print('price_x.transpose(1, 2).shape',price_x.transpose(1, 2).shape)
        price_y = self.p_tcn(price_x.transpose(1, 2)).transpose(1, 2)
#         print('price_y.shape',price_y.shape)
#         print('volume_x.transpose(1, 2).shape',volume_x.transpose(1, 2).shape)
        volume_y = self.v_tcn(volume_x.transpose(1, 2)).transpose(1, 2)
#         print('volume_y.shape',volume_y.shape)
        # Embedding
        mkt_x = self.mkt_embedding(mkt_x)
        crop_x = self.crop_embedding(crop_x)
        # wide component
        wide_x = torch.cat((price_y, volume_y), dim=1)
#         print('wide_x.transpose(1, 2)',wide_x.transpose(1, 2).shape)
        wide_y = self.wide_fc(wide_x.transpose(1, 2)).transpose(1, 2)
#         print('wide_y.shape',wide_y.shape)
#         wide_y = wide_y.reshape(wide_y.shape[0],-1,1)
#         print('wide_y.shape',wide_y.shape)
        # deep component
#         print('deep')

        mkt_x = mkt_x.view(mkt_x.shape[0],mkt_x.shape[1],1)
        geo_x = geo_x.view(geo_x.shape[0],geo_x.shape[1],1)
        crop_x = crop_x.view(crop_x.shape[0],crop_x.shape[1],1)
#         price_y = price_y.reshape(price_y.shape[0],price_y.shape[1])
#         volume_y = volume_y.reshape(volume_y.shape[0],volume_y.shape[1])
#         print('price_y.shape',price_y.shape)
#         print('volume_y.shape',volume_y.shape)
#         print('mkt_x.shape',mkt_x.shape)
#         print('crop_x.shape',crop_x.shape)
#         print('geo_x.shape',geo_x.shape)
        deep_x = torch.cat((price_y, volume_y, mkt_x, crop_x, geo_x), dim=1)
#         deep_x = deep_x.reshape(deep_x.shape[0],1,deep_x.shape[1])
#         print('deep_x.shape',deep_x.shape)
        deep_y = self.deep_fc(deep_x.transpose(1, 2)).transpose(1, 2)
#         deep_y = deep_y.reshape(deep_y.shape[0],-1,1)
#         print('deep_y.shape',deep_y.shape)
        # final fc
        final_x = torch.cat((wide_y, deep_y), dim=1)
        output = self.final_fc(final_x.transpose(1, 2))
        output = self.log_softmax(output.transpose(1, 2)).view(-1,2)
        return output
