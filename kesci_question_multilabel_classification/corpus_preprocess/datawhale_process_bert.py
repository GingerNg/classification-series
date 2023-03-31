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
from corpus_preprocess.encoders import BertEncoder

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
        text = str(text)
        text = en_basic_processor.lemmatize(text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        # remove punctuation and numbers
        regex = re.compile(
            '[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        nopunct = regex.sub(" ", text.lower())
        # return [token.text for token in tok.tokenizer(nopunct)]
        return nopunct.split()


processor = Processor()

encoder = BertEncoder()
# vocab_size = encoder.vocab_size
vocab_size = 100


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
    # bert encode
    label_list, text_list, token_type_ids_list, attention_mask_list = [], [], [], []
    for x, y in batch:
        label_list.append(label_encode(y))
        input_ids, token_type_ids, attention_mask = encoder.encode_ids(processor.process(x))
        text_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.tensor(text_list)
    type_ids = torch.tensor(token_type_ids_list)
    atten_masks = torch.tensor(attention_mask_list)
    return (text_list.to(device), type_ids.to(device), atten_masks.to(device)), label_list.to(device)


train_ds = DatawhaleDataset(X_train, y_train)
valid_ds = DatawhaleDataset(X_valid, y_valid)

batch_size = 64
# vocab_size = len(words)
train_dl = DataLoader(train_ds, batch_size=batch_size,
                      shuffle=True, collate_fn=collate_batch)
val_dl = DataLoader(valid_ds, batch_size=batch_size, collate_fn=collate_batch)
