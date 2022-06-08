import math
from typing import List, Optional

import torch
from torch import nn

from haystack.modeling.model.predictions import QACandidate
from torch.nn import CrossEntropyLoss

from haystack.modeling.model.prediction_head import QuestionAnsweringHead, FeedForwardBlock


class KentonQuestionAnsweringHead(QuestionAnsweringHead):
    def __init__(self, layer_dims: List[int] = [768, 2],
                 task_name: str = "question_answering",
                 no_ans_boost: float = 0.0,
                 context_window_size: int = 100,
                 n_best: int = 5,
                 n_best_per_sample: Optional[int] = None,
                 duplicate_filtering: int = -1,
                 temperature_for_confidence: float = 1.0,
                 use_confidence_scores_for_ranking: bool = False,
                 **kwargs):
        super(KentonQuestionAnsweringHead, self).__init__(layer_dims,
                                                          task_name,
                                                          no_ans_boost,
                                                          context_window_size,
                                                          n_best,
                                                          n_best_per_sample,
                                                          duplicate_filtering,
                                                          temperature_for_confidence,
                                                          use_confidence_scores_for_ranking,
                                                          **kwargs)

        self.mention_score = self.get_sequential(2 * layer_dims[0], 128, 1)
        self.pairwise_score = self.get_sequential(6 * layer_dims[0], 128, 1)

    @staticmethod
    def get_sequential(ind, hid, out):
        return nn.Sequential(
            nn.Linear(ind, hid),
            nn.ReLU(),
            nn.Linear(hid, out)
        )

    def forward(self, embedding: torch.Tensor, **kwargs):
        indicies = torch.arange(embedding.size(0))
        start_passage_idx = kwargs['seq_2_start_t'][0]
        query_ment_start = kwargs['query_ment_start'][0]
        query_ment_end = kwargs['query_ment_end'][0]

        # Query score, this time from the concat embedding (query, passage)
        query_start_embed = embedding[indicies, query_ment_start]
        query_end_embed = embedding[indicies, query_ment_end]
        g_query = torch.cat((query_start_embed, query_end_embed), dim=1)
        query_score = self.mention_score(g_query)

        # logits is of shape [batch_size, max_seq_len, 2]. Like above, the final dimension corresponds to [start, end]
        start_end_logits = self.feed_forward(embedding)
        start_logits, end_logits = start_end_logits.split(1, dim=-1)

        start_logits_clone = torch.clone(start_logits).squeeze(-1)
        end_logits_clone = torch.clone(end_logits).squeeze(-1)
        # remove query logits and leaving the [CLS] within the probability destirbution
        start_logits_clone[indicies, 1:start_passage_idx] = -math.inf
        end_logits_clone[indicies, 1:start_passage_idx] = -math.inf

        _, pass_start_idxs = torch.max(start_logits_clone, dim=1)
        _, pass_end_idxs = torch.max(end_logits_clone, dim=1)
        pass_start_embed = embedding[indicies, pass_start_idxs]
        pass_end_embed = embedding[indicies, pass_end_idxs]
        g_pass = torch.cat((pass_start_embed, pass_end_embed), dim=1)
        passage_score = self.mention_score(g_pass)

        g_query_pass = g_query * g_pass
        pairwise_feat = torch.cat((g_query, g_pass, g_query_pass), dim=-1)
        antecedent_score = self.pairwise_score(pairwise_feat)
        passage_logits = query_score + passage_score + antecedent_score

        # passage_logits = self.pass_selection(scores)

        return start_end_logits, passage_logits

    def logits_to_loss(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Combine predictions and labels to a per sample loss.
        """
        # todo explain how we only use first answer for train
        # labels.shape =  [batch_size, n_max_answers, 2]. n_max_answers is by default 6 since this is the
        # most that occurs in the SQuAD dev set. The 2 in the final dimension corresponds to [start, end]
        start_position = labels[:, 0, 0]
        end_position = labels[:, 0, 1]

        query_coref = kwargs['query_coref_link']
        passage_coref = kwargs['passage_coref_link']
        coref_labels = (query_coref == passage_coref).nonzero().squeeze(-1)
        # if coref_labels.size() == 0:
        #     coref_labels = torch.Tensor([start_position.size(0)], dtype=torch.int64, device=coref_labels.device)
        # coref_labels = torch.zeros(logits[1].size(), device=logits[1].device)
        # coref_labels[query_coref == passage_coref] = 1

        # logits is of shape [batch_size, max_seq_len, 2]. Like above, the final dimension corresponds to [start, end]
        start_logits, end_logits = logits[0].split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        passage_selection = logits[1]

        # Squeeze final singleton dimensions
        if len(start_position.size()) > 1:
            start_position = start_position.squeeze(-1)
        if len(end_position.size()) > 1:
            end_position = end_position.squeeze(-1)

        ignored_index = start_logits.size(1)
        start_position.clamp_(0, ignored_index)
        end_position.clamp_(0, ignored_index)

        # Workaround for pytorch bug in version 1.10.0 with non-continguous tensors
        # Fix expected in 1.10.1 based on https://github.com/pytorch/pytorch/pull/64954
        start_logits = start_logits.contiguous()
        start_position = start_position.contiguous()
        end_logits = end_logits.contiguous()
        end_position = end_position.contiguous()

        loss_fct = CrossEntropyLoss(reduction="mean")
        start_loss = loss_fct(start_logits, start_position)
        end_loss = loss_fct(end_logits, end_position)
        selection_loss = loss_fct(passage_selection.T, coref_labels)

        per_sample_loss = start_loss + end_loss + selection_loss
        return per_sample_loss

    def logits_to_preds(self, logits: torch.Tensor, span_mask: torch.Tensor, start_of_word: torch.Tensor,
                        seq_2_start_t: torch.Tensor, max_answer_length: int = 1000, **kwargs):
        """
        Get the predicted index of start and end token of the answer. Note that the output is at token level
        and not word level. Note also that these logits correspond to the tokens of a sample
        (i.e. special tokens, question tokens, passage_tokens)
        """

        # Will be populated with the top-n predictions of each sample in the batch
        # shape = batch_size x ~top_n
        # Note that ~top_n = n   if no_answer is     within the top_n predictions
        #           ~top_n = n+1 if no_answer is not within the top_n predictions
        all_top_n = []

        # logits is of shape [batch_size, max_seq_len, 2]. The final dimension corresponds to [start, end]
        start_logits, end_logits = logits[0].split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        passage_selection = logits[1]

        # Calculate a few useful variables
        batch_size = start_logits.size()[0]
        max_seq_len = start_logits.shape[1] # target dim

        # get scores for all combinations of start and end logits => candidate answers
        start_matrix = start_logits.unsqueeze(2).expand(-1, -1, max_seq_len)
        end_matrix = end_logits.unsqueeze(1).expand(-1, max_seq_len, -1)
        start_end_matrix = start_matrix + end_matrix

        # disqualify answers where end < start
        # (set the lower triangular matrix to low value, excluding diagonal)
        indices = torch.tril_indices(max_seq_len, max_seq_len, offset=-1, device=start_end_matrix.device)
        start_end_matrix[:, indices[0][:], indices[1][:]] = -888

        # disqualify answers where answer span is greater than max_answer_length
        # (set the upper triangular matrix to low value, excluding diagonal)
        indices_long_span = torch.triu_indices(max_seq_len, max_seq_len, offset=max_answer_length, device=start_end_matrix.device)
        start_end_matrix[:, indices_long_span[0][:], indices_long_span[1][:]] = -777

        # disqualify answers where start=0, but end != 0
        start_end_matrix[:, 0, 1:] = -666

        # Turn 1d span_mask vectors into 2d span_mask along 2 different axes
        # span mask has:
        #   0 for every position that is never a valid start or end index (question tokens, mid and end special tokens, padding)
        #   1 everywhere else
        span_mask_start = span_mask.unsqueeze(2).expand(-1, -1, max_seq_len)
        span_mask_end = span_mask.unsqueeze(1).expand(-1, max_seq_len, -1)
        span_mask_2d = span_mask_start + span_mask_end
        # disqualify spans where either start or end is on an invalid token
        invalid_indices = torch.nonzero((span_mask_2d != 2), as_tuple=True)
        start_end_matrix[invalid_indices[0][:], invalid_indices[1][:], invalid_indices[2][:]] = -999

        # Sort the candidate answers by their score. Sorting happens on the flattened matrix.
        # flat_sorted_indices.shape: (batch_size, max_seq_len^2, 1)
        flat_scores = start_end_matrix.view(batch_size, -1)
        flat_sorted_indices_2d = flat_scores.sort(descending=True)[1]
        flat_sorted_indices = flat_sorted_indices_2d.unsqueeze(2)

        # The returned indices are then converted back to the original dimensionality of the matrix.
        # sorted_candidates.shape : (batch_size, max_seq_len^2, 2)
        start_indices = flat_sorted_indices // max_seq_len
        end_indices = flat_sorted_indices % max_seq_len
        sorted_candidates = torch.cat((start_indices, end_indices), dim=2)

        # Get the n_best candidate answers for each sample
        for sample_idx in range(batch_size):
            sample_top_n = self.get_top_candidates(sorted_candidates[sample_idx],
                                                   start_end_matrix[sample_idx],
                                                   sample_idx,
                                                   start_matrix=start_matrix[sample_idx],
                                                   end_matrix=end_matrix[sample_idx],
                                                   passage_selection=passage_selection[sample_idx])
            all_top_n.append(sample_top_n)

        return all_top_n

    def get_top_candidates(self, sorted_candidates, start_end_matrix, sample_idx: int,
                           start_matrix, end_matrix, passage_selection):
        """
        Returns top candidate answers as a list of Span objects. Operates on a matrix of summed start and end logits.
        This matrix corresponds to a single sample (includes special tokens, question tokens, passage tokens).
        This method always returns a list of len n_best + 1 (it is comprised of the n_best positive answers along with the one no_answer)
        """
        # Initialize some variables
        top_candidates: List[QACandidate] = []
        n_candidates = sorted_candidates.shape[0]
        start_idx_candidates = set()
        end_idx_candidates = set()

        start_matrix_softmax_start = torch.softmax(start_matrix[:, 0], dim=-1)
        end_matrix_softmax_end = torch.softmax(end_matrix[0, :], dim=-1)
        # Iterate over all candidates and break when we have all our n_best candidates
        for candidate_idx in range(n_candidates):
            if len(top_candidates) == self.n_best_per_sample:
                break
            else:
                # Retrieve candidate's indices
                start_idx = sorted_candidates[candidate_idx, 0].item()
                end_idx = sorted_candidates[candidate_idx, 1].item()
                # Ignore no_answer scores which will be extracted later in this method
                if start_idx == 0 and end_idx == 0:
                    continue
                if self.duplicate_filtering > -1 and (start_idx in start_idx_candidates or end_idx in end_idx_candidates):
                    continue
                # score = start_end_matrix[start_idx, end_idx].item()
                score = passage_selection.item()
                confidence = (start_matrix_softmax_start[start_idx].item() + end_matrix_softmax_end[end_idx].item()) / 2
                top_candidates.append(QACandidate(offset_answer_start=start_idx,
                                                  offset_answer_end=end_idx,
                                                  score=score,
                                                  answer_type="span",
                                                  offset_unit="token",
                                                  aggregation_level="passage",
                                                  passage_id=str(sample_idx),
                                                  confidence=confidence))
                if self.duplicate_filtering > -1:
                    for i in range(0, self.duplicate_filtering + 1):
                        start_idx_candidates.add(start_idx + i)
                        start_idx_candidates.add(start_idx - i)
                        end_idx_candidates.add(end_idx + i)
                        end_idx_candidates.add(end_idx - i)

        # no_answer_score = start_end_matrix[0, 0].item()
        no_answer_score = -100
        no_answer_confidence = (start_matrix_softmax_start[0].item() + end_matrix_softmax_end[0].item()) / 2
        top_candidates.append(QACandidate(offset_answer_start=0,
                                          offset_answer_end=0,
                                          score=no_answer_score,
                                          answer_type="no_answer",
                                          offset_unit="token",
                                          aggregation_level="passage",
                                          passage_id=None,
                                          confidence=no_answer_confidence))

        return top_candidates
