import os
from abc import ABC
from pathlib import Path
from typing import Union

import torch
from haystack.modeling.model.language_model import LanguageModel, silence_transformers_logs
from transformers import BertModel, BertConfig


class WECEncoder(LanguageModel, ABC):
    def __init__(self):
        super(WECEncoder, self).__init__()
        self.device = "cpu"
        self.config = BertConfig.from_pretrained("SpanBERT/spanbert-base-cased")

    def set_tokenizer_size(self, token_len, device):
        self.model.resize_token_embeddings(token_len)
        self.device = device

    def segment_encode(self, input_ids, token_type_ids, input_mask, head_mask):
        if len(input_ids.size()) > 2:
            input_ids = torch.squeeze(input_ids)
            token_type_ids = torch.squeeze(token_type_ids)
            input_mask = torch.squeeze(input_mask)

        embedding_output = self.model.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        extended_attention_mask = input_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encodings = self.model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return encodings


class WECContextEncoder(WECEncoder):
    def __init__(self):
        super(WECContextEncoder, self).__init__()
        self.model = BertModel.from_pretrained('SpanBERT/spanbert-base-cased')
        self.name = "wec_context_encoder"

    def forward(self, passage_input_ids: torch.Tensor, passage_segment_ids: torch.Tensor,
                passage_attention_mask: torch.Tensor, **kwargs):
        head_mask = [None] * self.config.num_hidden_layers
        passage_encode = self.segment_encode(passage_input_ids,
                                             passage_segment_ids,
                                             passage_attention_mask,
                                             head_mask)

        # extract the last-hidden-state and then only the CLS token embeddings
        return passage_encode[0][:, 0, :], None

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

        head_mask = [None] * self.config.num_hidden_layers
        query_indices = torch.arange(start=0, end=query_input_ids.size(0), step=samples_size, device=self.device)
        query_input_ids_slc = torch.index_select(query_input_ids, dim=0, index=query_indices)
        query_segment_ids_slc = torch.index_select(query_segment_ids, dim=0, index=query_indices)
        query_input_mask_slc = torch.index_select(query_segment_ids, dim=0, index=query_indices)
        query_event_starts_slc = torch.index_select(query_event_starts, dim=0, index=query_indices)
        query_event_ends_slc = torch.index_select(query_event_ends, dim=0, index=query_indices)

        query_encode = self.segment_encode(query_input_ids_slc,
                                           query_segment_ids_slc,
                                           query_input_mask_slc,
                                           head_mask)

        out_query_embed = self.extract_query_embeddings(query_encode[0], query_event_starts_slc, query_event_ends_slc)
        return out_query_embed, None

    def freeze(self, layers):
        pass

    def unfreeze(self):
        pass

    @staticmethod
    def extract_query_embeddings(query_hidden_states,
                                 query_event_starts,
                                 query_event_ends):
        query_start_embeds = query_hidden_states[range(0, query_hidden_states.shape[0]), query_event_starts]
        query_end_embeds = query_hidden_states[range(0, query_hidden_states.shape[0]), query_event_ends]
        query_rep = (query_start_embeds * query_end_embeds) / 2
        return query_rep
