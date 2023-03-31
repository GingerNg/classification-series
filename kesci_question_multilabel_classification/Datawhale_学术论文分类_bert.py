import torch
# from corpus_preprocess.datawhale_process2 import train_dl, val_dl, vocab_size, LABELS
from corpus_preprocess.datawhale_process_bert import train_dl, val_dl, vocab_size, LABELS
# from models.text_lstm import LSTM
from models.bert import MyBert

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error


class Runner(object):
    def __init__(self, vocab_size, labels):
        super().__init__()
        # self.model = LSTM(vocab_size=vocab_size, num_tags=len(labels))
        self.model = MyBert(num_tags=len(labels))

    def run(self, mth='train'):
        if mth == "train":
            self.train_model()

    def train_model(self, epochs=1000, lr=0.001):
        parameters = filter(lambda p: p.requires_grad,
                            self.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=lr)
        for i in range(epochs):
            self.model.train()
            sum_loss = 0.0
            total = 0
            for x, y in train_dl:
                # x = x.long()
                # y = y.long()
                y_pred = self.model(x)
                optimizer.zero_grad()
                print(y_pred.shape, y.shape)
                loss = F.cross_entropy(y_pred, y)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()*y.shape[0]
                total += y.shape[0]
            # metric on train-data
            train_loss, train_acc, train_rmse = self.validation_metrics(train_dl)
            print("train loss %.3f, train accuracy %.3f, and train rmse %.3f" % (sum_loss/total, train_acc, train_rmse))
            val_loss, val_acc, val_rmse = self.validation_metrics(val_dl)
            # if i % 5 == 1:
            print("val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_acc, val_rmse))

    def validation_metrics(self, valid_dl):
        self.model.eval()
        correct = 0
        total = 0
        sum_loss = 0.0
        sum_rmse = 0.0
        for x, y in valid_dl:
            x = x.long()
            y = y.long()
            y_hat = self.model(x)
            loss = F.cross_entropy(y_hat, y)
            pred = torch.max(y_hat, 1)[1]
            correct += (pred == y).float().sum()
            total += y.shape[0]
            sum_loss += loss.item()*y.shape[0]
            sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
        return sum_loss/total, correct/total, sum_rmse/total


if __name__ == "__main__":
    runner = Runner(vocab_size=vocab_size, labels=LABELS)
    runner.run(mth="train")
