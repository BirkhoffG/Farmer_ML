import torch.nn as nn
import torch
import torch.optim as optim
from Model.TCN import TCN, WideTCN, TemporalConvNet
import time


class P1(nn.Module):
    def __init__(self, mkt_num, mkt_embedding_dim, num_channels, input_len, output_len, kernel_size=3, dropout=0.3):
        super(P1, self).__init__()
        self.tcn = TemporalConvNet(1, num_channels, kernel_size, dropout=dropout)
        self.mkt_embedding = nn.Embedding(mkt_num, mkt_embedding_dim)
        self.crop_embedding = nn.Embedding(6, 4)
        self.fc = nn.Linear(input_len + mkt_embedding_dim + 4, output_len)
        self.init_weights()

    def init_weights(self):
        # self.encoder.weight.data.normal_(0, 0.01)
        #         nn.init.xavier_normal_(self.tcn)
        nn.init.xavier_normal_(self.fc.weight.data)
        nn.init.normal_(self.fc.bias.data)
        nn.init.xavier_normal_(self.mkt_embedding.weight.data)
        nn.init.xavier_normal_(self.crop_embedding.weight.data)

    def forward(self, price_x, volume_x, mkt_x, crop_x):
        price_y = self.p_tcn(price_x.transpose(1, 2)).transpose(1, 2)
        embedded_mkt_x = self.mkt_embedding(mkt_x).transpose(1, 2)
        embedded_crop_x = self.crop_embedding(crop_x).transpose(1, 2)
        y = torch.cat((price_y, embedded_mkt_x, embedded_crop_x), dim=1)
        return self.fc(y)

    def fit(self, lr, epochs, train_loader, val_loader, train_std, train_mean, log):
        # loss function
        criterion = nn.MSELoss()
        # optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        # track loss list
        loss_list, rmse_list, val_loss_list, val_rmse_list = tuple([] for _ in range(4))
        total_steps = len(train_loader)

        for epoch in range(epochs):
            # start training
            self.train()
            #         poly_lr_scheduler(optimizer, lr, epoch+1)

            for ix, (price_train_x, volume_train_x, mkt_train_x, crop_train_x, y) in enumerate(
                    train_loader):
                # track starting time
                start_time = time.time()

                # clear the accumulated gradient before each instance
                self.zero_grad()

                # prepare the data and label
                price_train_x, volume_train_x, mkt_train_x, crop_train_x, y = \
                    to_device(price_train_x, volume_train_x, mkt_train_x, crop_train_x, y)

                # run the forward pass
                outputs = self(price_train_x.float(), volume_train_x.float(), mkt_train_x.long(), crop_train_x.long())

                # Compute the loss, gradients, and update the parameters
                loss = criterion(outputs, y.float())

                # back propogation
                loss.backward()
                # update parameters
                optimizer.step()
                # clear gradient
                optimizer.zero_grad()

                # append loss to the loss list
                rmse = RMSE(outputs, y.float(), train_std, train_mean)
                rmse_list.append(rmse)
                loss_list.append(loss)

                # print in every 50 episodes
                if (ix + 1) % 50 == 0:
                    log.info(f'Epoch [{epoch + 1}/{epochs}], Step [{ix + 1}/{total_steps}], '
                             f'Time [{time.time() - start_time:.6f} sec], Avg loss: {sum(loss_list[-50:]) / 50: .4f}, '
                             f'Avg RMSE: {sum(rmse_list[-50:]) / 50: .4f}')


p1 = P1(mkt_num=800, mkt_embedding_dim=128, num_channels=[16, 8, 4, 1], input_len=90,
        output_len=1, kernel_size=3, dropout=0.3)
