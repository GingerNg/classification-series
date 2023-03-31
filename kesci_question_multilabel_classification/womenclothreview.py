import torch
# from corpus_preprocess.womenclothreview_process import train_dl, val_dl, vocab_size, zero_numbering
from corpus_preprocess.womenclothreview_process2 import train_dl, val_dl, vocab_size, zero_numbering
from models.text_lstm import LSTM
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

class Runner(object):
    def __init__(self):
        super().__init__()

    def run(self):
        pass

def train_model(model, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y in train_dl:
            # print(x)
            x = x.long()
            y = y.long()
            y_pred = model(x)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        # metric on train-data
        train_loss, train_acc, train_rmse = validation_metrics(model, train_dl)
        print("train loss %.3f, train accuracy %.3f, and train rmse %.3f" % (sum_loss/total, train_acc, train_rmse))
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        # if i % 5 == 1:
        print("val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_acc, val_rmse))


def validation_metrics(model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total

model = LSTM(vocab_size=vocab_size, num_tags=len(zero_numbering))
train_model(model, epochs=30, lr=0.01)
