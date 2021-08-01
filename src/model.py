import torch
from torch import nn
from transformers import BertForQuestionAnswering, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertEmbeddings, BertPooler


class WecEsModel(nn.Module):
    def __init__(self, tokenizer_len):
        super(WecEsModel, self).__init__()
        self.cfg = BertConfig.from_pretrained("external/spanbert_hf_base")
        self.embeddings = BertEmbeddings(self.cfg)
        self.query_encode = BertEncoder(self.cfg)
        self.passege_encoder = BertEncoder(self.cfg)
        self.spanbert = BertForQuestionAnswering.from_pretrained('external/spanbert_hf_base')
        self.spanbert.resize_token_embeddings(tokenizer_len)

    def forward(self, passage_input_ids, query_input_ids,
                passage_input_mask, query_input_mask,
                passage_segment_ids, query_segment_ids,
                passage_start_position, passage_end_position):
        concat_input = torch.cat((passage_input_ids, query_input_ids), dim=1)
        concat_mask = torch.cat((passage_input_mask, query_input_mask), dim=1)
        concat_segments = torch.cat((passage_segment_ids, query_segment_ids), dim=1)

        return self.spanbert(concat_input, concat_segments, concat_mask, output_hidden_states=False,
                             start_positions=passage_start_position, end_positions=passage_end_position)

    def forward_new(self, passage_input_ids, query_input_ids,
                            passage_input_mask, query_input_mask,
                            passage_segment_ids, query_segment_ids,
                            passage_start_position, passage_end_position):

        passage_encode = self.passage_encode(passage_input_ids,
                                             passage_input_mask,
                                             passage_segment_ids)

        query_encode = self.query_encode(query_input_ids,
                                         query_input_mask,
                                         query_segment_ids)

        passage_sequence_output = passage_encode[0]
        query_sequence_output = query_encode[0]
        concat_input = torch.cat((passage_sequence_output, query_sequence_output), dim=1)

        # return self.spanbert(concat_input, token_type_ids, attention_mask, output_hidden_states=False,
        #                      start_positions=passage_start_position, end_positions=passage_end_position)

    def query_encode(self, input_ids, attention_mask, token_type_ids):
        device = input_ids.device
        input_shape = input_ids.size()
        head_mask = self.spanbert.get_head_mask(None, self.cfg.num_hidden_layers)
        extended_attention_mask: torch.Tensor = self.spanbert.get_extended_attention_mask(attention_mask, input_shape, device)
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        query_encode = self.query_encode(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return query_encode

    def passage_encode(self, input_ids, attention_mask, token_type_ids):
        device = input_ids.device
        input_shape = input_ids.size()
        head_mask = self.spanbert.get_head_mask(None, self.cfg.num_hidden_layers)
        extended_attention_mask: torch.Tensor = self.spanbert.get_extended_attention_mask(attention_mask, input_shape, device)
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        passage_encode = self.passege_encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return passage_encode
