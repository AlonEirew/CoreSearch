import torch
from torch import nn
from transformers import BertForQuestionAnswering, BertTokenizer


class WecEsModel(object):
    def __init__(self, hidden_size):
        self.tokenizer = BertTokenizer.from_pretrained('spanbert-base-cased')
        self.spanbert = BertForQuestionAnswering.from_pretrained('spanbert-base-cased')
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.add_tokens()

    def add_tokens(self):
        self.tokenizer.add_tokens(["[QSPAN_START]"])
        self.tokenizer.add_tokens(["[QSPAN_END]"])
        self.spanbert.resize_token_embeddings(len(self.tokenizer))
        self.spanbert.embeddings.word_embeddings.weight[-1, :] = torch.zeros([self.spanbert.config.hidden_size])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, _ = self.spanbert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits
