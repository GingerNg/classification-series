# token2ids
from torch import nn
# from nlp_tools.tokenizers import WhitespaceTokenizer
from transformers import BertTokenizer


class BertEncoder(nn.Module):
    def __init__(self, bert_path=None, maxlen=200):
        super(BertEncoder, self).__init__()
        # self.tokenizer = WhitespaceTokenizer(bert_path)
        # self.slow_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.maxlen = maxlen

    def encode_ids(self, tokens):
        try:
            # tokens = self.tokenizer.tokenize(tokens)
            tokens = tokens[0: self.maxlen]
            tokenized_text = ["CLS"] + tokens + ['SEP']
            # print(tokenized_text)
            input_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
            token_type_ids = [0] * len(input_ids)
            attention_mask = [1] * len(input_ids)

            padding_length = self.maxlen + 2 - len(input_ids)
            if padding_length > 0:  # pad
                input_ids = input_ids + ([0] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)
            return input_ids, token_type_ids, attention_mask
        except Exception as e:
            print(tokens)
            raise Exception(e.__repr__())

    def encode(self, tokens):
        tokens = self.tokenizer.tokenize(tokens)
        return tokens


if __name__ == "__main__":
    bert_encoder = BertEncoder()
    print(bert_encoder.encode_ids("i am ok"))