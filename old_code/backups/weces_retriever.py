import torch
import torch.nn.functional as F
from torch import nn

from src.override_classes.retriever.wec_encoders import WECQuestionEncoder, WECContextEncoder


class WECESRetriever(nn.Module):
    def __init__(self, query_model, passage_model, token_len, device):
        super(WECESRetriever, self).__init__()
        self.query_encoder = WECQuestionEncoder()
        self.query_encoder.set_tokenizer_size(token_len, device)

        self.passage_encoder = WECContextEncoder()
        self.passage_encoder.set_tokenizer_size(token_len, device)

        self.device = device

    def forward(self, passage_input_ids, query_input_ids,
                passage_input_mask, query_input_mask,
                passage_segment_ids, query_segment_ids, **kwargs):
        samples_size = 1
        if "sample_size" in kwargs:
            samples_size = kwargs["sample_size"]

        # The input ids before arrive with the number of (positive + negatives) at dim=1 (batch_size, samples, max_passage_len)
        passage_input_ids = passage_input_ids.view(passage_input_ids.shape[0] * samples_size, -1)
        passage_segment_ids = passage_segment_ids.view(passage_segment_ids.shape[0] * samples_size, -1)
        passage_input_mask = passage_input_mask.view(passage_input_mask.shape[0] * samples_size, -1)
        passage_pos_indices = torch.arange(start=0, end=passage_input_ids.size(0), step=samples_size, device=self.device)

        passage_encode, _ = self.passage_encoder(passage_input_ids,
                                                 passage_segment_ids,
                                                 passage_input_mask,
                                                 **kwargs)

        query_encode, _ = self.query_encoder(query_input_ids,
                                             query_input_mask,
                                             query_segment_ids,
                                             **kwargs)

        # shape_dim0 = passage_encode.shape[0]
        # passage_seq_to_expls = passage_encode.view(int(shape_dim0 / samples_size), samples_size, -1)
        loss, softmax_scores = self.predict(query_encode, passage_encode, passage_pos_indices)
        return loss, softmax_scores, passage_pos_indices

    def predict(self, query_rep, passage_rep, passage_pos_indices):
        # positive_idxs = torch.zeros(query_rep.shape[0], dtype=torch.long)
        softmax_scores = self.predict_softmax(query_rep, passage_rep)
        loss = F.nll_loss(
            softmax_scores,
            passage_pos_indices,
            reduction="mean",
        )
        return loss, softmax_scores

    @staticmethod
    def predict_softmax(query_rep, passage_rep):
        # sim_batch = torch.matmul(query_rep.unsqueeze(1), torch.transpose(passage_rep, 1, 2))
        sim_batch = torch.matmul(query_rep, torch.transpose(passage_rep, 0, 1))
        softmax_scores = F.log_softmax(sim_batch.squeeze(1), dim=1)
        return softmax_scores
