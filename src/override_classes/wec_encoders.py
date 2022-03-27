from pathlib import Path

import torch
import transformers
from haystack.modeling.model.language_model import DPRContextEncoder, DPRQuestionEncoder


class WECContextEncoder(DPRContextEncoder):
    def __init__(self):
        super(WECContextEncoder, self).__init__()
        # self.dropout = nn.Dropout(0.1)

    def forward(self, passage_input_ids: torch.Tensor, passage_segment_ids: torch.Tensor,
                passage_attention_mask: torch.Tensor, **kwargs):

        if len(passage_input_ids.size()) > 2:
            max_seq_len = passage_input_ids.shape[-1]
            passage_input_ids = passage_input_ids.view(-1, max_seq_len)
            passage_segment_ids = passage_segment_ids.view(-1, max_seq_len)
            passage_attention_mask = passage_attention_mask.view(-1, max_seq_len)

        passage_encode = self.model(input_ids=passage_input_ids,
                                    token_type_ids=passage_segment_ids,
                                    attention_mask=passage_attention_mask,
                                    output_attentions=False,
                                    output_hidden_states=True)

        # extract the last-hidden-state CLS token embeddings
        # return self.dropout(passage_encode.pooler_output), None
        return passage_encode.hidden_states[-1][:, 0, :], None
        # return self.dropout(passage_encode.last_hidden_state[:, 0, :]), passage_encode.attentions
        # return passage_encode.pooler_output, None

    def save_config(self, save_dir):
        save_filename = Path(save_dir) / "language_model_config.json"
        with open(save_filename, "w") as file:
            setattr(self.model.config, "name", self.__class__.__name__)
            setattr(self.model.config, "language", self.language)
            # For DPR models, transformers overwrites the model_type with the one set in DPRConfig
            # Therefore, we copy the model_type from the model config to DPRConfig
            if self.__class__.__name__ == "WECQuestionEncoder" or self.__class__.__name__ == "WECContextEncoder":
                setattr(transformers.DPRConfig, "model_type", self.model.config.model_type)
            string = self.model.config.to_json_string()
            file.write(string)

    def freeze(self, layers):
        pass

    def unfreeze(self):
        pass


class WECQuestionEncoder(DPRQuestionEncoder):
    def __init__(self):
        super(WECQuestionEncoder, self).__init__()
        # self.dropout = nn.Dropout(0.1)

    def forward(self, query_input_ids: torch.Tensor, query_segment_ids: torch.Tensor,
                query_attention_mask: torch.Tensor, **kwargs):
        query_event_starts = None
        query_event_ends = None
        if "query_start" in kwargs and "query_end" in kwargs:
            query_event_starts = kwargs["query_start"]
            query_event_ends = kwargs["query_end"]

        # query_indices = torch.arange(start=0, end=query_input_ids.size(0), step=samples_size, device=self.device)
        # query_input_ids_slc = torch.index_select(query_input_ids, dim=0, index=query_indices)
        # query_segment_ids_slc = torch.index_select(query_segment_ids, dim=0, index=query_indices)
        # query_input_mask_slc = torch.index_select(query_segment_ids, dim=0, index=query_indices)
        query_encode = self.model(input_ids=query_input_ids,
                                  token_type_ids=query_segment_ids,
                                  attention_mask=query_attention_mask,
                                  output_attentions=False,
                                  output_hidden_states=True)

        last_hidden = query_encode.hidden_states[-1]
        if query_event_starts is not None and query_event_ends is not None:
            # Will trigger while training
            # query_start_embeds = last_hidden[range(0, last_hidden.shape[0]), query_event_starts]
            # query_end_embeds = last_hidden[range(0, last_hidden.shape[0]), query_event_ends]
            # query_rep = (query_start_embeds + query_end_embeds) / 2
            # return query_rep, None
            return last_hidden[:, 0, :], None
        else:
            # Will trigger at inference
            # return self.dropout(query_encode.pooler_output), None
            return last_hidden[:, 0, :], None
            # return query_encode.pooler_output, None

    def save_config(self, save_dir):
        save_filename = Path(save_dir) / "language_model_config.json"
        with open(save_filename, "w") as file:
            setattr(self.model.config, "name", self.__class__.__name__)
            setattr(self.model.config, "language", self.language)
            # For DPR models, transformers overwrites the model_type with the one set in DPRConfig
            # Therefore, we copy the model_type from the model config to DPRConfig
            if self.__class__.__name__ == "WECQuestionEncoder" or self.__class__.__name__ == "WECContextEncoder":
                setattr(transformers.DPRConfig, "model_type", self.model.config.model_type)
            string = self.model.config.to_json_string()
            file.write(string)

    def freeze(self, layers):
        pass

    def unfreeze(self):
        pass

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
