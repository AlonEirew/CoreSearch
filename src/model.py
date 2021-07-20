from torch import nn
from transformers import BertForQuestionAnswering


class WecEsModel(nn.Module):
    def __init__(self, tokenizer_len):
        super(WecEsModel, self).__init__()
        self.spanbert = BertForQuestionAnswering.from_pretrained('external/spanbert_hf_base')
        self.spanbert.resize_token_embeddings(tokenizer_len)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,  start_positions=0, end_positions=0):
        return self.spanbert(input_ids, token_type_ids, attention_mask, output_hidden_states=False,
                             start_positions=start_positions, end_positions=end_positions)
