
from utils import file_utils, model_utils
import torch.optim as optim
import torch
from corpus_preprocess.datawhale_process3 import vocab_size, train_iter, val_iter, LABELS
import torch.nn.functional as F
from models.text_lstm import LSTM
from evaluation_index.metric_factory import get_metric
from evaluation_index.utils import reformat
import torch
EarlyStopEpochs = 3

metric_obj = get_metric(name='prec_rec_f1')
# metric_obj = get_metric(name='acc')


class Runner(object):
    def __init__(self, vocab_size, labels):
        super().__init__()
        self.model = LSTM(vocab_size, len(labels))

    def _eval(self, data):
        self.model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch_labels = batch.label
                torch.cuda.empty_cache()
                batch_input = batch.title.transpose(0, 1)
                batch_outputs = self.model(batch_input)
                y_pred.extend(torch.argmax(
                    batch_outputs, dim=1).cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())
            score, dev_f1 = metric_obj.get_score(y_true, y_pred)
        return score, dev_f1

    def _infer(self, data):
        self.model.eval()
        # # data = dev_data
        # y_pred = []
        # with torch.no_grad():
        #     for batch_data in data_iter(data, test_batch_size, shuffle=False):
        #         torch.cuda.empty_cache()
        #         batch_inputs, batch_labels = batch2tensor(batch_data)
        #         batch_outputs = model(batch_inputs)
        #         y_pred.extend(torch.max(batch_outputs, dim=1)
        #                         [1].cpu().numpy().tolist())
        #         print(label_encoder.label2name(y_pred))

    def run(self, method="train", save_path=None, infer_texts=[]):

        step = 0
        if method == "train":
            self.model.train()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          self.model.parameters()),
                                   lr=0.001)
            loss_funtion = F.cross_entropy

            epochs = 10
            best_dev_f1 = 0.0
            for epoch in range(epochs):
                overall_losses = 0
                y_pred = []
                y_true = []
                for step, batch in enumerate(train_iter):
                    optimizer.zero_grad()
                    batch_labels = batch.label
                    # print(batch_labels.shape)
                    batch_input = batch.text.transpose(0, 1)
                    # print(batch_input)
                    batch_outputs = self.model(batch_input)
                    loss = loss_funtion(batch_outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                    loss_value = loss.detach().cpu().item()
                    overall_losses += loss_value
                    # print(batch_outputs)
                    # print(torch.argmax(batch_outputs, dim=1))
                    # print(batch_labels)
                    y_pred.extend(torch.argmax(batch_outputs, dim=1).cpu().numpy().tolist())
                    y_true.extend(batch_labels.cpu().numpy().tolist())

                overall_losses /= train_iter.batch_size
                overall_losses = reformat(overall_losses, 4)
                score, train_f1 = metric_obj.get_score(y_true, y_pred)
                print("epoch:{}, {}:{}, train_f1:{}".format(epoch,
                                                            metric_obj.name(),
                                                            train_f1,
                                                            score))

                _, dev_f1 = self._eval(data=val_iter)

                if best_dev_f1 <= dev_f1:
                    best_dev_f1 = dev_f1
                    early_stop = 0
                    best_train_f1 = train_f1
                    save_path = model_utils.save_checkpoint(
                        self.model, epoch, save_folder="./data/datawhale")
                    print("save_path:{}".format(save_path))
                else:
                    early_stop += 1
                    if early_stop == EarlyStopEpochs:
                        break

                print("score:{}, dev_f1:{}, best_train_f1:{}, best_dev_f1:{}, overall_loss:{}".format(
                    dev_f1, score, best_train_f1, best_dev_f1, overall_losses
                ))


if __name__ == "__main__":
    runner = Runner(vocab_size=vocab_size, labels=LABELS)
    runner.run(method="train")
