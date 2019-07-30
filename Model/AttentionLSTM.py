import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True, layer_num=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.layerNum = layer_num

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, batch_first=True)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        h0 = torch.zeros(self.layerNum * 1, batch_size, self.hidden_size)
        c0 = torch.zeros(self.layerNum * 1, batch_size, self.hidden_size)
        h0, c0 = h0.to(device), c0.to(device)

        output, hidden = self.lstm(inputs, (h0, c0))
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size),
                torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size))


class AttentionDecoder(nn.Module):

    def __init__(self, hidden_size, output_size, vocab_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.attn = nn.Linear(hidden_size + output_size, 1)
        self.lstm = nn.LSTM(hidden_size + vocab_size, output_size)
        self.final = nn.Linear(output_size, vocab_size)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.output_size),
                torch.zeros(1, 1, self.output_size))

    def forward(self, encoder_outputs, input):
        # encode_output = [batch_size * 2, input_len, hidden_size]

        weights = []
        batch_size = input.size(0)
        decoder_hidden = (torch.zeros(1, batch_size, self.hidden_size).to(device),
                          torch.zeros(1, batch_size, self.hidden_size).to(device))

        for i in range(len(encoder_outputs)):
            print(f"decoder_hidden[0][0].shape: {decoder_hidden[0][0].shape}")
            print(f"encoder_outputs[0].shape: {encoder_outputs[0].shape}")
            weights.append(self.attn(torch.cat((decoder_hidden[0][0],
                                                encoder_outputs[i]), dim=1)))
        normalized_weights = F.softmax(torch.cat(weights, 1), 1)

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                                 encoder_outputs.view(1, -1, self.hidden_size))

        input_lstm = torch.cat((attn_applied[0], input[0]), dim=1) #if we are using embedding, use embedding of input here instead

        output, hidden = self.lstm(input_lstm.unsqueeze(0), decoder_hidden)

        output = self.final(output[0])

        return output, hidden, normalized_weights


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, output_size, output_dim):
        super(AttentionLSTM, self).__init__()
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional)
        self.decoder = AttentionDecoder(hidden_size=hidden_size*(1+int(bidirectional)),
                                        output_size=output_size, vocab_size=output_dim)

    def forward(self, inputs):
        # encode_output = [batch_size, input_len, hidden_size]
        # encode_hidden = [input_size, batch_size, hidden_size]
        encoder_output, encoded_hidden = self.encoder(inputs)
        print(f"encode_output shape: {encoder_output.size()}")
        print(f"encode_hidden shape: {encoded_hidden[0].size()}")
        print(f"decoder encoder_outputs shape: {torch.cat((encoder_output, encoder_output)).size()}")

        # encode_output = [batch_size * 2, input_len, hidden_size]
        output, decoder_hidden, normalized_weights = \
            self.decoder(torch.cat((encoder_output, encoder_output)), inputs)
        return output


model = AttentionLSTM(input_size=1, hidden_size=8,
                      bidirectional=False, output_size=1, output_dim=1)

#%%
x = torch.rand(size=(64, 90, 1))

y = model(x)
