import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence


class LSTM(nn.Module):

    def __init__(self, vocab_size, num_tags):
        super(LSTM, self).__init__()
        self.bidir = False
        self.hidden_dim = 128
        self.word_embeddings = nn.Embedding(vocab_size, 300)  # embedding之后的shape: torch.Size([200, 8, 300])
        # embedding.weight.data.copy_(weight_matrix)
        # self.word_embeddings.requires_grad = False
        if self.bidir:
            self.lstm = nn.LSTM(input_size=300, hidden_size=128, bidirectional=True, batch_first=True)  # torch.Size([200, 8, 128])
            self.fc1 = nn.Linear(self.hidden_dim*2, self.hidden_dim*2)
            self.fc2 = nn.Linear(self.hidden_dim*2, num_tags)
        else:
            self.lstm = nn.LSTM(input_size=300, hidden_size=128, bidirectional=False, batch_first=True)  # torch.Size([200, 8, 128])
            self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, num_tags)

    def forward(self, sentence):
        # print(sentence)
        embeds = self.word_embeddings(sentence)
        # print(embeds)
        _, (hidden, cell) = self.lstm(embeds)
        # print(hidden.shape)
        if self.bidir:
            lstm_out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            lstm_out = hidden[-1]
        # print(lstm_out.shape)
        # final = F.relu(lstm_out)  # 8*128
        y = F.softmax(self.fc2(self.fc1(lstm_out)), dim=-1)
        # print(y.shape)
        return y


class LSTM_variable_input(nn.Module):
    def __init__(self, vocab_size, num_tags):
        super().__init__()
        hidden_dim = 128
        embedding_dim = 300
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_tags)

    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out


# from torch import nn
# import torch
# import torch.nn.functional as F
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# # https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0

# class LSTM(nn.Module):

#     def __init__(self, vocab=[], dimension=128):
#         super(LSTM, self).__init__()

#         self.embedding = nn.Embedding(len(vocab), 300)
#         self.dimension = dimension
#         self.lstm = nn.LSTM(input_size=300,
#                             hidden_size=dimension,
#                             num_layers=1,
#                             batch_first=True,
#                             bidirectional=True)
#         self.drop = nn.Dropout(p=0.5)

#         self.fc = nn.Linear(2*dimension, 1)

#     def forward(self, text, text_len):

#         text_emb = self.embedding(text)

#         packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
#         packed_output, _ = self.lstm(packed_input)
#         output, _ = pad_packed_sequence(packed_output, batch_first=True)

#         out_forward = output[range(len(output)), text_len - 1, :self.dimension]
#         out_reverse = output[:, 0, self.dimension:]
#         out_reduced = torch.cat((out_forward, out_reverse), 1)
#         text_fea = self.drop(out_reduced)

#         text_fea = self.fc(text_fea)
#         text_fea = torch.squeeze(text_fea, 1)
#         text_out = torch.sigmoid(text_fea)

#         return text_out