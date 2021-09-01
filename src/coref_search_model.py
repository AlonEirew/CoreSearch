import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertModel

from src.data_obj import QuestionAnsweringModelOutput


class SimilarityModel(object):
    def __init__(self, in_batch_samples, device):
        self.in_batch_samples = in_batch_samples
        self.device = device

    @staticmethod
    def extract_passage_embeddings(outputs: QuestionAnsweringModelOutput, passages_bounds):
        for i in range(outputs.start_logits.shape[0]):
            outputs.start_logits[i][passages_bounds[i]:] = -math.inf
            outputs.end_logits[i][passages_bounds[i]:] = -math.inf

        softmax_starts = torch.softmax(outputs.start_logits, dim=1)
        softmax_ends = torch.softmax(outputs.end_logits, dim=1)
        start_idxs, end_idxs = torch.argmax(softmax_starts, dim=1), torch.argmax(softmax_ends, dim=1)
        indicies = torch.arange(outputs.passage_hidden_states.size(0))
        passages_start_embeds = outputs.passage_hidden_states[indicies, start_idxs]
        passage_end_embeds = outputs.passage_hidden_states[indicies, end_idxs]
        passage_rep = torch.cat((passages_start_embeds, passage_end_embeds), dim=1)
        return passage_rep

    def extract_query_embeddings(self, query_hidden_states, query_event_starts, query_event_ends,
                                 tot_exmp_per_query):
        # +1 for positive example
        query_indices = torch.arange(start=0, end=query_hidden_states.size(0), step=tot_exmp_per_query)
        query_event_starts = torch.index_select(query_event_starts, dim=0, index=query_indices)
        query_event_ends = torch.index_select(query_event_ends, dim=0, index=query_indices)
        if tot_exmp_per_query > 1:
            query_hidden_avg = torch.mean(query_hidden_states.view(self.in_batch_samples, tot_exmp_per_query,
                                                                   query_hidden_states.shape[1], -1), dim=1)
        else:
            query_hidden_avg = query_hidden_states

        query_div_inds = torch.arange(query_hidden_avg.size(0))
        query_start_embeds = query_hidden_states[query_div_inds, query_event_starts]
        query_end_embeds = query_hidden_states[query_div_inds, query_event_ends]
        # in_batch_samples, neg + pos for each query, concat embeddings of start + end)
        # passage_rep = torch.cat((passages_start_embeds, passage_end_embeds), dim=1)
        query_rep = torch.cat((query_start_embeds, query_end_embeds), dim=1)
        return query_rep

    def calc_similarity_loss(self, query_rep, passage_rep):
        positive_idxs = torch.zeros(self.in_batch_samples, dtype=torch.long)
        predicted_idxs, softmax_scores = self.predict_softmax(query_rep, passage_rep)
        sim_loss = F.nll_loss(
            softmax_scores,
            positive_idxs.to(softmax_scores.device),
            reduction="mean",
        )
        return sim_loss, predicted_idxs

    def predict_softmax(self, query_rep, passage_rep):
        sim_batch = self.get_similarity_score(query_rep, passage_rep)
        softmax_scores = F.log_softmax(sim_batch, dim=1)
        _, predicted_idxs = torch.max(softmax_scores, 1)
        return predicted_idxs, softmax_scores

    def get_similarity_score(self, query_rep, passage_rep):
        sim_batch = torch.tensor([]).to(self.device)
        for i, query_emb in enumerate(query_rep):
            if passage_rep.size() != query_rep.size():
                sim_q_p = torch.matmul(query_emb.view(1, -1), torch.transpose(passage_rep[i], 0, 1))
            else:
                sim_q_p = torch.matmul(query_emb.view(1, -1), passage_rep.view(-1, 1))
            sim_batch = torch.cat([sim_batch, sim_q_p], dim=0)
        return sim_batch

    @staticmethod
    def predict_pairwise(query_rep, passage_rep):
        prediction = torch.cosine_similarity(query_rep, passage_rep)
        prediction = torch.round(prediction)
        return prediction


class SpanPredAuxiliary(nn.Module):
    def __init__(self, tokenizer_len):
        super(SpanPredAuxiliary, self).__init__()
        self.cfg = BertConfig.from_pretrained("external/spanbert_hf_base")
        self.qa_outputs = nn.Linear(self.cfg.hidden_size, self.cfg.num_labels)

        self.query_spanbert = BertModel.from_pretrained('external/spanbert_hf_base')
        self.query_spanbert.resize_token_embeddings(tokenizer_len)
        self.passage_spanbert = BertModel.from_pretrained('external/spanbert_hf_base')

    def forward(self, passage_input_ids, query_input_ids,
                            passage_input_mask, query_input_mask,
                            passage_segment_ids, query_segment_ids,
                            start_positions, end_positions):
        head_mask = [None] * self.cfg.num_hidden_layers

        passage_encode = self.segment_encode(self.passage_spanbert,
                                             passage_input_ids,
                                             passage_segment_ids,
                                             passage_input_mask,
                                             head_mask)

        query_encode = self.segment_encode(self.query_spanbert,
                                           query_input_ids,
                                           query_segment_ids,
                                           query_input_mask,
                                           head_mask)

        passage_sequence_output = passage_encode[0]
        query_sequence_output = query_encode[0]
        concat_encode_output = torch.cat((passage_sequence_output, query_sequence_output), dim=1)

        total_loss, start_logits, end_logits = self.get_qa_answer(concat_encode_output, start_positions, end_positions)

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            query_hidden_states=query_sequence_output,
            passage_hidden_states=passage_sequence_output,
            start_logits=start_logits,
            end_logits=end_logits
        )

    @staticmethod
    def segment_encode(model, input_ids, token_type_ids, query_input_mask, head_mask):
        embedding_output = model.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        extended_attention_mask = query_input_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        query_encode = model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return query_encode

    def get_qa_answer(self, encoder_output, start_positions, end_positions):
        logits = self.qa_outputs(encoder_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss, start_logits, end_logits
