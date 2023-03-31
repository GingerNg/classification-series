# build vocab
from collections import Counter
# from transformers import BasicTokenizer
# import logging
import numpy as np


class BaikeVocab():
    def __init__(self, vocab_path):
        super().__init__()
        self.unk = -1
        self._id2word = []
        self.build_vocab(vocab_path)
        def reverse(x): return dict(zip(x, range(len(x))))  # 词与id的映射
        self._word2id = reverse(self._id2word)  # dict， word: id

    def build_vocab(self, vocab_path):
        lines = open(vocab_path, "r").readlines()
        words = []
        for line in lines:
            words.append(line.strip())
        # print(words)
        self._id2word = words

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)


# basic_tokenizer = BasicTokenizer()
class EmbVocab():  # 词向量词典
    def __init__(self, embfile):
        super().__init__()
        self.pad = 0
        self.unk = 1
        self._id2extword = ['[PAD]', '[UNK]']
        self.word_dim = 0
        self.word_count = 0
        self.embeddings = self.load_pretrained_embs(embfile=embfile)

    def load_pretrained_embs(self, embfile):
        """[summary]

        Args:
            embfile ([type]): [description]

        Returns:
            [list]: [返回词向量， 词向量的索引=词在词表中的id]
        """
        with open(embfile, encoding='utf-8') as f:
            lines = f.readlines()
            items = lines[0].split()
            word_count, embedding_dim = int(items[0]), int(items[1])
        self.word_count = word_count
        self.word_dim = embedding_dim

        # 词向量读入内存
        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        for line in lines[1:]:
            values = line.split()
            self._id2extword.append(values[0])
            vector = np.array(values[1:], dtype='float64')
            embeddings[self.unk] += vector  # 未知词向量＝已知词向量的平均
            embeddings[index] = vector
            index += 1

        embeddings[self.unk] = embeddings[self.unk] / word_count

        embeddings = embeddings / np.std(embeddings)

        def reverse(x): return dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        assert len(set(self._id2extword)) == len(self._id2extword)

        return embeddings

    @property
    def extword_size(self):
        return len(self._id2extword)

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)


#　词表
class Vocab():
    def __init__(self, train_data):
        self.min_count = 5
        self.pad = 0
        self.unk = 1
        self._id2word = ['[PAD]', '[UNK]']  # list， 词在list中的下标==词ID
        self._id2extword = ['[PAD]', '[UNK]']

        self._id2label = []
        self.target_names = []

        self.build_vocab(train_data)

        def reverse(x): return dict(zip(x, range(len(x))))  # 词与id的映射
        self._word2id = reverse(self._id2word)  # dict， word: id
        self._label2id = reverse(self._id2label)

        # logging.info("Build vocab: words %d, labels %d." %
        #              (self.word_size, self.label_size))

    def build_vocab(self, data):
        """
        根据文本数据生成词表
        """
        self.word_counter = Counter()

        # process text
        for text in data['text']:
            words = text.split()
            for word in words:
                self.word_counter[word] += 1

        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        # # process label
        # label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政',
        #               5: '社会', 6: '教育', 7: '财经', 8: '家居', 9: '游戏',
        #               10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}

        # self.label_counter = Counter(data['label'])

        # for label in range(len(self.label_counter)):
        #     count = self.label_counter[label]
        #     self._id2label.append(label)
        #     self.target_names.append(label2name[label])

    # def load_pretrained_embs(self, embfile):
    #     """[summary]

    #     Args:
    #         embfile ([type]): [description]

    #     Returns:
    #         [list]: [返回词向量， 词向量的索引=词在词表中的id]
    #     """
    #     with open(embfile, encoding='utf-8') as f:
    #         lines = f.readlines()
    #         items = lines[0].split()
    #         word_count, embedding_dim = int(items[0]), int(items[1])

    #     # 词向量读入内存
    #     index = len(self._id2extword)
    #     embeddings = np.zeros((word_count + index, embedding_dim))
    #     for line in lines[1:]:
    #         values = line.split()
    #         self._id2extword.append(values[0])
    #         vector = np.array(values[1:], dtype='float64')
    #         embeddings[self.unk] += vector  # 未知词向量＝已知词向量的平均
    #         embeddings[index] = vector
    #         index += 1

    #     embeddings[self.unk] = embeddings[self.unk] / word_count

    #     embeddings = embeddings / np.std(embeddings)

    #     def reverse(x): return dict(zip(x, range(len(x))))
    #     self._extword2id = reverse(self._id2extword)

    #     assert len(set(self._id2extword)) == len(self._id2extword)

    #     return embeddings

    def word2id(self, xs):  # 根据word获取id
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)

    # def extword2id(self, xs):
    #     if isinstance(xs, list):
    #         return [self._extword2id.get(x, self.unk) for x in xs]
    #     return self._extword2id.get(xs, self.unk)

    # def label2id(self, xs):
    #     if isinstance(xs, list):
    #         return [self._label2id.get(x, self.unk) for x in xs]
    #     return self._label2id.get(xs, self.unk)

    @property
    def word_size(self):  # 词表中词汇个数
        return len(self._id2word)

    # @property
    # def extword_size(self):
    #     return len(self._id2extword)

    # @property
    # def label_size(self):
    #     return len(self._id2label)


# vocab = Vocab(train_data)
