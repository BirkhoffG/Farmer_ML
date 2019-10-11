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

        # TODO: adjust initialization method
        # self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.final_fc.weight.data)
        nn.init.normal_(self.final_fc.bias.data)

    def forward(self, price_x, volume_x, geo_x, mkt_x, crop_x):
        # TCN
        price_y = self.p_tcn(price_x.transpose(1, 2)).transpose(1, 2)
        volume_y = self.v_tcn(volume_x.transpose(1, 2)).transpose(1, 2)
        # Embedding
        mkt_x = self.mkt_embedding(mkt_x)
        crop_x = self.crop_embedding(crop_x)
        # wide component
        wide_x = torch.cat((price_y, volume_y), dim=1)
        wide_y = self.wide_fc(wide_x)
        # deep component
        deep_x = torch.cat((price_y, volume_y, mkt_x, crop_x, geo_x), dim=1)
        deep_y = self.deep_fc(deep_x)
        # final fc
        final_x = torch.cat((deep_x, deep_y), dim=1)

        # print(f"price_y shape: {price_y.size()}; volume_y shape: {volume_y.size()}")
        # print(f"y shape: {y.size()}")
        # print("y.size: ", y.size())
        return self.final_fc(final_x)
