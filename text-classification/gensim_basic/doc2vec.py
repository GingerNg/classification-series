
# -*- coding: utf-8 -*-
"""
https://radimrehurek.com/gensim/models/doc2vec.html
"""
import sys
import logging
import os
import pandas as pd
import gensim
# 引入doc2vec
import jieba
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import common_texts

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
# from utilties import ko_title2words

# 引入日志配置
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def segment(content):
    segs = jieba.cut(content.replace("\n",""))  # 默认方式分词，分词结果用空格隔开
    # segs = filter(lambda x: len(x) > 1, segs)  # 去掉长度小于1的词
    # segs = filter(lambda x: x not in stopWordList, segs)  # 去掉停用词
    return list(segs)

# 加载数据
documents = []
# 使用count当做每个句子的“标签”，标签和每个句子是一一对应的
count = 0

corpus_data = pd.read_csv("./file.csv", encoding="utf-8", sep=",")
lines = corpus_data.text.values.tolist()

# with open('../data/titles/ko.video.corpus' ,'r') as f:
for line in lines:
    # 切词，返回的结果是列表类型
    words = segment(line)
    # 这里documents里的每个元素是二元组，具体可以查看函数文档
    documents.append(gensim.models.doc2vec.TaggedDocument(words, [str(count)]))
    count += 1
    if count % 10 == 0:
        logging.info('{} has loaded...'.format(count))

# 模型训练
model = Doc2Vec(documents, dm=1, size=100, window=8, min_count=5, workers=4)
# 保存模型
model.save('./ko_d2v.model')

# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
# model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
# model = doc2vec.Doc2Vec.load('models/ko_d2v.model')

words = u"控诉“勇刚猛”三兄弟 兰世立打落麦趣尔股价"
vec = model.infer_vector(segment(words))
print(vec)