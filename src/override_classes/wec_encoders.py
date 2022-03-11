from abc import ABC

import torch
from haystack.modeling.model.language_model import LanguageModel
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

        encodings = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        return encodings


class WECContextEncoder(WECEncoder):
    def __init__(self):
        super(WECContextEncoder, self).__init__()
        self.model = BertModel.from_pretrained('SpanBERT/spanbert-base-cased')
        self.name = "wec_context_encoder"

    def forward(self, passage_input_ids: torch.Tensor, passage_segment_ids: torch.Tensor,
                passage_attention_mask: torch.Tensor, **kwargs):
        passage_encode = self.segment_encode(passage_input_ids,
                                             passage_segment_ids,
                                             passage_attention_mask)

        # extract the last-hidden-state CLS token embeddings
        return passage_encode[0][:, 0, :], None
        # return passage_encode.pooler_output, None

    def freeze(self, layers):
        pass

    def unfreeze(self):
        pass


class WECQuestionEncoder(WECEncoder):
    def __init__(self):
        super(WECQuestionEncoder, self).__init__()
        self.model = BertModel.from_pretrained('SpanBERT/spanbert-base-cased')
        self.name = "wec_quesion_encoder"

    def forward(self, query_input_ids: torch.Tensor, query_segment_ids: torch.Tensor,
                query_attention_mask: torch.Tensor, **kwargs):
        query_event_starts = kwargs["query_start"]
        query_event_ends = kwargs["query_end"]
        samples_size = 1
        if "sample_size" in kwargs:
            samples_size = kwargs["sample_size"]

        query_indices = torch.arange(start=0, end=query_input_ids.size(0), step=samples_size, device=self.device)
        query_input_ids_slc = torch.index_select(query_input_ids, dim=0, index=query_indices)
        query_segment_ids_slc = torch.index_select(query_segment_ids, dim=0, index=query_indices)
        query_input_mask_slc = torch.index_select(query_segment_ids, dim=0, index=query_indices)
        query_event_starts_slc = torch.index_select(query_event_starts, dim=0, index=query_indices)
        query_event_ends_slc = torch.index_select(query_event_ends, dim=0, index=query_indices)

        query_encode = self.segment_encode(query_input_ids_slc,
                                           query_segment_ids_slc,
                                           query_input_mask_slc)

        out_query_embed = self.extract_query_span_embeddings(query_encode[0], query_event_starts_slc, query_event_ends_slc)
        return out_query_embed, None
        # return query_encode[0][:, 0, :], None

    def freeze(self, layers):
        pass

    def unfreeze(self):
        pass

    @staticmethod
    def extract_query_startend_embeddings(query_hidden_states,
                                          query_event_starts,
                                          query_event_ends):
        query_start_embeds = query_hidden_states[range(0, query_hidden_states.shape[0]), query_event_starts]
        query_end_embeds = query_hidden_states[range(0, query_hidden_states.shape[0]), query_event_ends]
        query_rep = (query_start_embeds * query_end_embeds) / 2
        return query_rep

    @staticmethod
    def extract_query_span_embeddings(query_hidden_states,
                                      query_event_starts,
                                      query_event_ends):
        # query_span = query_hidden_states[range(0, query_hidden_states.shape[0]), torch.cat((query_event_starts.view(-1,1), query_event_ends.view(-1,1)), dim=1)]
        # result = torch.empty(query_hidden_states.size(0), query_hidden_states.size(2))
        result = list()
        for i in range(query_hidden_states.size(0)):
            query_span = query_hidden_states[0][query_event_starts[0]:query_event_ends[0]]
            result.append(torch.mean(query_span, dim=0))
        return torch.stack(result)
