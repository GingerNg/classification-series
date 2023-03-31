from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import re
import string
import pandas as pd
from utils.constants import HOST_IP
import pymongo
from collections import Counter
from sklearn.model_selection import train_test_split
from utils.model_utils import device
from nlp_tools.basic_process import EnProcessor

en_basic_processor = EnProcessor()

col_label = 'categories'
dataset_name = 'Datawhale_学术论文分类_数据集'
client = pymongo.MongoClient(HOST_IP, 27017)
db = client[dataset_name]
col = db["train_data"]
df = pd.DataFrame(list(col.find()))
df['text'] = df['title'] + ' ' + df['abstract']

# df = df[0:1000]  # 小批量验证

LABELS = list(set(df[col_label].to_list()))
print(LABELS)
print(len(LABELS))

X = list(df['text'])
y = list(df['categories'])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)


class Processor(object):
    def __init__(self):
        super().__init__()

    def process(self, text):
        text = en_basic_processor.lemmatize(text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        # remove punctuation and numbers
        regex = re.compile(
            '[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        nopunct = regex.sub(" ", text.lower())
        # return [token.text for token in tok.tokenizer(nopunct)]
        return nopunct.split()


class Encoder(object):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def encode_ids(self, tokenized):
        N = 200  # seq_len
        encoded = np.zeros(N, dtype=int)
        enc1 = np.array([self.vocab.get(word, self.vocab["UNK"])
                        for word in tokenized])
        length = min(N, len(enc1))
        encoded[:length] = enc1[:length]
        return encoded


processor = Processor()
# count number of occurences of each word
counts = Counter()
for index, row in df.iterrows():
    counts.update(processor.process(row['text']))

# deleting infrequent words
print("num_words before:", len(counts.keys()))
for word in list(counts):
    if counts[word] < 2:
        del counts[word]
print("num_words after:", len(counts.keys()))

vocab2index = {"": 0, "UNK": 1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)

encoder = Encoder(vocab=vocab2index)
vocab_size = encoder.vocab_size


class DatawhaleDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def label_encode(y): return LABELS.index(y)


def collate_batch(batch):
    label_list, text_list = [], []
    for x, y in batch:
        label_list.append(label_encode(y))
        text_list.append(encoder.encode_ids(processor.process(x)))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.tensor(text_list)
    return text_list.to(device), label_list.to(device)


train_ds = DatawhaleDataset(X_train, y_train)
valid_ds = DatawhaleDataset(X_valid, y_valid)

batch_size = 64
# vocab_size = len(words)
train_dl = DataLoader(train_ds, batch_size=batch_size,
                      shuffle=True, collate_fn=collate_batch)
val_dl = DataLoader(valid_ds, batch_size=batch_size, collate_fn=collate_batch)
