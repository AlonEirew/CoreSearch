import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, BertModel


class WECESRetriever(nn.Module):
    def __init__(self, tokenizer_len, device):
        super(WECESRetriever, self).__init__()
        self.cfg = BertConfig.from_pretrained("SpanBERT/spanbert-base-cased")
        self.query_encoder = BertModel.from_pretrained('SpanBERT/spanbert-base-cased')
        self.query_encoder.resize_token_embeddings(tokenizer_len)
        self.passage_encoder = BertModel.from_pretrained('SpanBERT/spanbert-base-cased')
        self.device = device

    def forward(self, passage_input_ids, query_input_ids,
                passage_input_mask, query_input_mask,
                passage_segment_ids, query_segment_ids,
                query_event_starts, query_event_ends, samples_size):
        head_mask = [None] * self.cfg.num_hidden_layers

        passage_encode = self.segment_encode(self.passage_encoder,
                                             passage_input_ids,
                                             passage_segment_ids,
                                             passage_input_mask,
                                             head_mask)

        query_indices = torch.arange(start=0, end=query_input_ids.size(0), step=samples_size, device=self.device)
        query_input_ids_slc = torch.index_select(query_input_ids, dim=0, index=query_indices)
        query_segment_ids_slc = torch.index_select(query_segment_ids, dim=0, index=query_indices)
        query_input_mask_slc = torch.index_select(query_input_mask, dim=0, index=query_indices)
        query_event_starts_slc = torch.index_select(query_event_starts, dim=0, index=query_indices)
        query_event_ends_slc = torch.index_select(query_event_ends, dim=0, index=query_indices)

        query_encode = self.segment_encode(self.query_encoder,
                                           query_input_ids_slc,
                                           query_segment_ids_slc,
                                           query_input_mask_slc,
                                           head_mask)

        # extract the last-hidden-state and then only the CLS token embeddings
        passage_sequence_output = passage_encode[0][:, 0, :]
        query_sequence_output = self.extract_query_embeddings(query_encode[0],
                                                              query_event_starts_slc,
                                                              query_event_ends_slc)

        shape_dim0 = passage_sequence_output.shape[0]
        passage_seq_to_expls = passage_sequence_output.view(int(shape_dim0 / samples_size), samples_size, -1)
        return self.predict(query_sequence_output, passage_seq_to_expls)

    def predict(self, query_rep, passage_rep):
        positive_idxs = torch.zeros(query_rep.shape[0], dtype=torch.long)
        predicted_idxs, softmax_scores = self.predict_softmax(query_rep, passage_rep)
        sim_loss = F.nll_loss(
            softmax_scores,
            positive_idxs.to(softmax_scores.device),
            reduction="mean",
        )
        return sim_loss, predicted_idxs

    @staticmethod
    def predict_softmax(query_rep, passage_rep):
        sim_batch = torch.matmul(query_rep.unsqueeze(1), torch.transpose(passage_rep, 1, 2))
        softmax_scores = F.log_softmax(sim_batch.squeeze(1), dim=1)
        _, predicted_idxs = torch.max(softmax_scores, 1)
        return predicted_idxs, softmax_scores

    @staticmethod
    def predict_pairwise(query_rep, passage_rep):
        prediction = torch.cosine_similarity(query_rep, passage_rep)
        prediction = torch.round(prediction)
        return prediction

    @staticmethod
    def segment_encode(model, input_ids, token_type_ids, input_mask, head_mask):
        embedding_output = model.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        extended_attention_mask = input_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encodings = model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return encodings

    @staticmethod
    def extract_query_embeddings(query_hidden_states,
                                 query_event_starts,
                                 query_event_ends):
        query_start_embeds = query_hidden_states[range(0, query_hidden_states.shape[0]), query_event_starts]
        query_end_embeds = query_hidden_states[range(0, query_hidden_states.shape[0]), query_event_ends]
        query_rep = (query_start_embeds * query_end_embeds) / 2
        return query_rep
