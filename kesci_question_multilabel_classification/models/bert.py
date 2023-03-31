from nlp_tools.tokenizers import WhitespaceTokenizer
from torch import nn
import logging
from transformers import BertModel, BertForPreTraining
from models.attentions import Attention
from models.encoders import SentEncoder
from utils.model_utils import use_cuda, device
import numpy as np
from cfg import bert_path
import torch.nn.functional as F

# build word encoder
dropout = 0.15


class MyBert(nn.Module):
    def __init__(self, num_tags):
        super(MyBert, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder.requires_grad = False
        self.out = nn.Linear(768, num_tags, bias=True)


    def forward(self, batch_inputs):
        # batch_inputs(batch_inputs1, batch_inputs2): b x doc_len x sent_len
        # batch_masks : b x doc_len x sent_len
        input_ids, token_type_ids, attention_mask = batch_inputs
        bert_output = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        # print(embedding)
        batch_outputs = self.out(bert_output.pooler_output)
        batch_outputs = F.softmax(batch_outputs, dim=-1)
        return batch_outputs
