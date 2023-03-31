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



###
dataset_name = 'demo_dataset'
client = pymongo.MongoClient(HOST_IP, 27017)
db = client[dataset_name]
col = db["Womens Clothing E-Commerce Reviews"]
df = pd.DataFrame(list(col.find()))[0:1000]

df['Title'] = df['Title'].fillna('')
df['Review Text'] = df['Review Text'].fillna('')
df['review'] = df['Title'] + ' ' + df['Review Text']

df = df[['review', 'Rating']]
df.columns = ['review', 'rating']
df['review_length'] = df['review'].apply(lambda x: len(x.split()))

#changing ratings to 0-numbering
zero_numbering = {1:0, 2:1, 3:2, 4:3, 5:4}
df['rating'] = df['rating'].apply(lambda x: zero_numbering[x])


class Processor(object):
    def __init__(self):
        super().__init__()

    def process(self, text):
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
        nopunct = regex.sub(" ", text.lower())
        # return [token.text for token in tok.tokenizer(nopunct)]
        return nopunct.split()


class Encoder(object):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def encode_ids(self, tokenized):
        N = 70
        encoded = np.zeros(N, dtype=int)
        enc1 = np.array([self.vocab.get(word, self.vocab["UNK"]) for word in tokenized])
        length = min(N, len(enc1))
        encoded[:length] = enc1[:length]
        return encoded

processor = Processor()

#count number of occurences of each word
counts = Counter()
for index, row in df.iterrows():
    counts.update(processor.process(row['review']))

#deleting infrequent words
print("num_words before:", len(counts.keys()))
for word in list(counts):
    if counts[word] < 2:
        del counts[word]
print("num_words after:", len(counts.keys()))


# 构建词典
vocab2index = {"": 0, "UNK": 1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)

encoder = Encoder(vocab=vocab2index)

df['encoded'] = df['review'].apply(lambda x: np.array(encoder.encode_ids(processor.process(x))))

X = list(df['encoded'])
y = list(df['rating'])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

vocab_size = encoder.vocab_size

class DatawhaleDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].astype(np.int32)), self.y[idx]


train_ds = DatawhaleDataset(X_train, y_train)
valid_ds = DatawhaleDataset(X_valid, y_valid)

batch_size = 5000
# vocab_size = len(words)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size)

