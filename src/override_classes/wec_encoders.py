from abc import ABC

import torch
from haystack.modeling.model.language_model import LanguageModel
from torch import nn
from transformers import BertModel, BertConfig


class WECEncoder(LanguageModel, ABC):
    def __init__(self):
        super(WECEncoder, self).__init__()
        self.device = "cpu"
        self.config = BertConfig.from_pretrained("SpanBERT/spanbert-base-cased")

    def set_tokenizer_size(self, token_len, device):
        self.model.resize_token_embeddings(token_len)
        self.device = device

    def segment_encode(self, input_ids, token_type_ids, input_mask):
        if len(input_ids.size()) > 2:
            input_ids = torch.squeeze(input_ids)
            token_type_ids = torch.squeeze(token_type_ids)
            input_mask = torch.squeeze(input_mask)

        encodings = self.model(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=input_mask,
                               output_attentions=True)
        return encodings


class WECContextEncoder(WECEncoder):
    def __init__(self, model):
        super(WECContextEncoder, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.dropout = nn.Dropout(0.1)
        self.name = "wec_context_encoder"

    def forward(self, passage_input_ids: torch.Tensor, passage_segment_ids: torch.Tensor,
                passage_attention_mask: torch.Tensor, **kwargs):
        passage_encode = self.segment_encode(passage_input_ids,
                                             passage_segment_ids,
                                             passage_attention_mask)

        if len(kwargs) > 1:
            # extract the last-hidden-state CLS token embeddings
            # return self.dropout(passage_encode[0][:, 0, :]), None
            return passage_encode.last_hidden_state[:, 0, :], passage_encode.attentions
            # return passage_encode.pooler_output, None
        else:
            return passage_encode.last_hidden_state[:, 0, :], passage_encode.attentions

    def freeze(self, layers):
        pass

    def unfreeze(self):
        pass


class WECQuestionEncoder(WECEncoder):
    def __init__(self, model):
        super(WECQuestionEncoder, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.dropout = nn.Dropout(0.1)
        self.name = "wec_quesion_encoder"

    def forward(self, query_input_ids: torch.Tensor, query_segment_ids: torch.Tensor,
                query_attention_mask: torch.Tensor, **kwargs):
        query_event_starts = None
        query_event_ends = None
        if "query_start" in kwargs and "query_end" in kwargs:
            query_event_starts = kwargs["query_start"]
            query_event_ends = kwargs["query_end"]
        samples_size = 1
        if "sample_size" in kwargs:
            samples_size = kwargs["sample_size"]

        query_indices = torch.arange(start=0, end=query_input_ids.size(0), step=samples_size, device=self.device)
        query_input_ids_slc = torch.index_select(query_input_ids, dim=0, index=query_indices)
        query_segment_ids_slc = torch.index_select(query_segment_ids, dim=0, index=query_indices)
        query_input_mask_slc = torch.index_select(query_segment_ids, dim=0, index=query_indices)
        query_encode = self.segment_encode(query_input_ids_slc,
                                           query_segment_ids_slc,
                                           query_input_mask_slc)

        if query_event_starts is not None and query_event_ends is not None:
            # Will trigger while training
            # query_event_starts_slc = torch.index_select(query_event_starts, dim=0, index=query_indices)
            # query_event_ends_slc = torch.index_select(query_event_ends, dim=0, index=query_indices)
            # out_query_embed = self.extract_query_start_end_embeddings(query_encode[0], query_event_starts_slc, query_event_ends_slc)
            # return out_query_embed, None
            # return self.dropout(query_encode[0][:, 0, :]), Noneֿ
            return query_encode.last_hidden_state[:, 0, :], query_encode.attentions
            # return query_encode.pooler_output, None
        else:
            # Will trigger at inference
            return query_encode.last_hidden_state[:, 0, :], query_encode.attentions
            # return query_encode.pooler_output, None

    def freeze(self, layers):
        pass

    def unfreeze(self):
        pass

    @staticmethod
    def extract_query_start_end_embeddings(query_hidden_states,
                                           query_event_starts,
                                           query_event_ends):
        query_start_embeds = query_hidden_states[range(0, query_hidden_states.shape[0]), query_event_starts]
        query_end_embeds = query_hidden_states[range(0, query_hidden_states.shape[0]), query_event_ends]
        query_rep = query_start_embeds * query_end_embeds
        return query_rep

    # This method expects no spacial tokens (QUERY_START/QUERY_END)
    @staticmethod
    def extract_query_span_embeddings(query_hidden_states,
                                      query_event_starts,
                                      query_event_ends):
        # query_span = query_hidden_states[range(0, query_hidden_states.shape[0]), torch.cat((query_event_starts.view(-1,1), query_event_ends.view(-1,1)), dim=1)]
        # result = torch.empty(query_hidden_states.size(0), query_hidden_states.size(2))
        result = list()
        for i in range(query_hidden_states.size(0)):
            query_span = query_hidden_states[0][query_event_starts[0]:query_event_ends[0]+1]
            result.append(torch.mean(query_span, dim=0))
        return torch.stack(result)
