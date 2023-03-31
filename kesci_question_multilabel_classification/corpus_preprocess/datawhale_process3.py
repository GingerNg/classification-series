from torchtext.data import Iterator, BucketIterator
import pandas as pd
import torch
from torchtext.data import Dataset
from tqdm import tqdm
from torchtext import data
from utils.constants import HOST_IP
import pandas as pd
import pymongo
import numpy as np
from utils.model_utils import device
import random


class DatawhaleDataset(Dataset):
    name = 'Datawhale Dataset'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, df, text_field, label_field, test=False, aug=False, **kwargs):
        fields = [("id", None),
                  ("text", text_field), ("label", label_field)]

        examples = []

        if test:
            # 如果为测试集，则不加载label
            for text in tqdm(df['text']):
                examples.append(data.Example.fromlist(
                    [None, text, None], fields))
        else:
            for text, label in tqdm(zip(df['text'], df[col_label])):
                label = LABELS.index(label)
                if aug:
                    # do augmentation
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout(text)
                    else:
                        text = self.shuffle(text)
                examples.append(data.Example.fromlist(
                    [None, text, label], fields))
        super(DatawhaleDataset, self).__init__(examples, fields)

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self, text, p=0.5):
        # random delete some text
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)


### demo_dataset
# col_label = 'label'
# dataset_name = "demo_dataset"
# client = pymongo.MongoClient(HOST_IP, 27017)
# db = client[dataset_name]
# col = db["spam_ham_dataset"]
# df = pd.DataFrame(
#     list(col.find({"$or": [{"label": 'spam'}, {"label": 'ham'}]})))

###
col_label = 'categories'
dataset_name = 'Datawhale_学术论文分类_数据集'
client = pymongo.MongoClient(HOST_IP, 27017)
db = client[dataset_name]
col = db["train_data"]
df = pd.DataFrame(list(col.find()))
df['text'] = df['title'] + ' ' + df['abstract']

df = df[0:1000]
# print(df.groupby('categories').size())
LABELS = list(set(df[col_label].to_list()))
print(LABELS)
print(len(LABELS))

data_df = df.sample(frac=1).reset_index(drop=True)
cut_off = int(len(data_df)*.8)
train_df = data_df[:cut_off]
val_df = data_df[cut_off:]


X = list(df['review'])
y = list(df['rating'])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)


def tokenize(x): return str(x).lower().replace("\n", " ").split()


TEXT = data.Field(sequential=True, tokenize=tokenize, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False)

train = DatawhaleDataset(train_df, text_field=TEXT,
                         label_field=LABEL, test=False)
valid = DatawhaleDataset(val_df, text_field=TEXT,
                         label_field=LABEL, test=False)

TEXT.build_vocab(train)
vocab_size = len(TEXT.vocab)
print(len(TEXT.vocab))

# 若只针对训练集构造迭代器
# train_iter = data.BucketIterator(dataset=train, batch_size=8, shuffle=True, sort_within_batch=False, repeat=False)

# 同时对训练集和验证集进行迭代器的构建
train_iter, val_iter = BucketIterator.splits(
    (train, valid),
    batch_sizes=(8, 8),
    device=device,
    # the BucketIterator needs to be told what function it should use to group the data.
    sort_key=lambda x: len(x.text),
    sort_within_batch=False,
    repeat=False,
    shuffle=(True, False)
)