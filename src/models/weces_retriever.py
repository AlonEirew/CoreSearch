import torch
import torch.nn.functional as F
from torch import nn

from src.override_classes.wec_encoders import WECQuestionEncoder, WECContextEncoder


class WECESRetriever(nn.Module):
    def __init__(self, query_model, passage_model, token_len, device):
        super(WECESRetriever, self).__init__()
        self.query_encoder = WECQuestionEncoder(query_model)
        self.query_encoder.set_tokenizer_size(token_len, device)

        self.passage_encoder = WECContextEncoder(passage_model)
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

        passage_encode, _ = self.passage_encoder(passage_input_ids,
                                                 passage_segment_ids,
                                                 passage_input_mask,
                                                 **kwargs)

        query_encode, _ = self.query_encoder(query_input_ids,
                                             query_input_mask,
                                             query_segment_ids,
                                             **kwargs)

        shape_dim0 = passage_encode.shape[0]
        passage_seq_to_expls = passage_encode.view(int(shape_dim0 / samples_size), samples_size, -1)
        return self.predict(query_encode, passage_seq_to_expls)

    def predict(self, query_rep, passage_rep):
        positive_idxs = torch.zeros(query_rep.shape[0], dtype=torch.long)
        predicted_idxs, softmax_scores = self.predict_softmax(query_rep, passage_rep)
        loss = F.nll_loss(
            softmax_scores,
            positive_idxs.to(softmax_scores.device),
            reduction="mean",
        )
        return loss, predicted_idxs

    @staticmethod
    def predict_softmax(query_rep, passage_rep):
        sim_batch = torch.matmul(query_rep.unsqueeze(1), torch.transpose(passage_rep, 1, 2))
        softmax_scores = F.log_softmax(sim_batch.squeeze(1), dim=1)
        _, predicted_idxs = torch.max(softmax_scores, 1)
        return predicted_idxs, softmax_scores

    @staticmethod
    def predict_pairwise_cosine(query_rep, passage_rep):
        prediction = torch.cosine_similarity(query_rep, passage_rep)
        # prediction = torch.round(prediction)
        return prediction

    @staticmethod
    def predict_pairwise_dot_product(query_rep, passage_rep):
        # prediction = torch.dot(query_rep, passage_rep)
        prediction = query_rep @ passage_rep.T
        # prediction = torch.round(prediction)
        return prediction
