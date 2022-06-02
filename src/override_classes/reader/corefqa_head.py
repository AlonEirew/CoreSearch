import json
import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple

import torch
from haystack.modeling.data_handler.samples import SampleBasket
from haystack.modeling.model.prediction_head import PredictionHead, FeedForwardBlock
from haystack.modeling.model.predictions import QACandidate, QAPred
from haystack.modeling.utils import try_get
from torch import nn, optim
from torch.nn import CrossEntropyLoss, NLLLoss
from transformers import AutoModelForQuestionAnswering

logger = logging.getLogger(__name__)


class CorefQuestionAnsweringHead(PredictionHead):
    """
    A question answering head predicts the start and end of the answer on token level.

    In addition, it gives a score for the prediction so that multiple answers can be ranked.
    There are three different kinds of scores available:
    1) (standard) score: the sum of the logits of the start and end index. This score is unbounded because the logits are unbounded.
    It is the default for ranking answers.
    2) confidence score: also based on the logits of the start and end index but scales them to the interval 0 to 1 and incorporates no_answer.
    It can be used for ranking by setting use_confidence_scores_for_ranking to True
    3) calibrated confidence score: same as 2) but divides the logits by a learned temperature_for_confidence parameter
    so that the confidence scores are closer to the model's achieved accuracy. It can be used for ranking by setting
    use_confidence_scores_for_ranking to True and temperature_for_confidence!=1.0. See examples/question_answering_confidence.py for more details.
    """

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
        """
        :param layer_dims: dimensions of Feed Forward block, e.g. [768,2], for adjusting to BERT embedding. Output should be always 2
        :param kwargs: placeholder for passing generic parameters
        :param no_ans_boost: How much the no_answer logit is boosted/increased.
                             The higher the value, the more likely a "no answer possible given the input text" is returned by the model
        :param context_window_size: The size, in characters, of the window around the answer span that is used when displaying the context around the answer.
        :param n_best: The number of positive answer spans for each document.
        :param n_best_per_sample: num candidate answer spans to consider from each passage. Each passage also returns "no answer" info.
                                  This is decoupled from n_best on document level, since predictions on passage level are very similar.
                                  It should have a low value
        :param duplicate_filtering: Answers are filtered based on their position. Both start and end position of the answers are considered.
                                    The higher the value, answers that are more apart are filtered out. 0 corresponds to exact duplicates. -1 turns off duplicate removal.
        :param temperature_for_confidence: The divisor that is used to scale logits to calibrate confidence scores
        :param use_confidence_scores_for_ranking: Whether to sort answers by confidence score (normalized between 0 and 1) or by standard score (unbounded)(default).
        """
        super(CorefQuestionAnsweringHead, self).__init__()
        if len(kwargs) > 0:
            logger.warning(f"Some unused parameters are passed to the QuestionAnsweringHead. "
                           f"Might not be a problem. Params: {json.dumps(kwargs)}")
        self.layer_dims = layer_dims
        assert self.layer_dims[-1] == 2
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        logger.debug(f"Prediction head initialized with size {self.layer_dims}")
        self.num_labels = self.layer_dims[-1]
        self.ph_output_type = "per_token_squad"
        self.model_type = ("span_classification")  # predicts start and end token of answer
        self.task_name = task_name
        self.no_ans_boost = no_ans_boost
        self.context_window_size = context_window_size
        self.n_best = n_best
        if n_best_per_sample:
            self.n_best_per_sample = n_best_per_sample
        else:
            # increasing n_best_per_sample to n_best ensures that there are n_best predictions in total
            # otherwise this might not be the case for very short documents with only one "sample"
            self.n_best_per_sample = n_best
        self.duplicate_filtering = duplicate_filtering
        self.generate_config()
        self.temperature_for_confidence = nn.Parameter(torch.ones(1) * temperature_for_confidence)
        self.use_confidence_scores_for_ranking = use_confidence_scores_for_ranking
        # Dim is the contatenation of the (mention_start & mention end) * 2 + (multiplication of them)
        self.start_end_ff = FeedForwardBlock([2 * layer_dims[0], 1])
        self.pairwize = FeedForwardBlock([2 * layer_dims[0], 1])
        # self.linear = nn.Linear(128, 1)

    @staticmethod
    def get_sequential(ind, out):
        return nn.Sequential(
            nn.Linear(ind, out),
            nn.ReLU()
        )

    @classmethod
    def load(cls, pretrained_model_name_or_path: Union[str, Path], revision: Optional[str] = None,
             **kwargs):  # type: ignore
        """
        Load a prediction head from a saved Haystack or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a Haystack prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g. distilbert-base-uncased-distilled-squad)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - distilbert-base-uncased-distilled-squad
                                              - bert-large-uncased-whole-word-masking-finetuned-squad

                                              See https://huggingface.co/models for full list
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        """
        if os.path.exists(pretrained_model_name_or_path) \
                and "config.json" in str(pretrained_model_name_or_path) \
                and "prediction_head" in str(pretrained_model_name_or_path):
            # a) Haystack style
            super(CorefQuestionAnsweringHead, cls).load(str(pretrained_model_name_or_path))
        else:
            # b) transformers style
            # load all weights from model
            full_qa_model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path,
                                                                          revision=revision, **kwargs)
            # init empty head
            head = cls(layer_dims=[full_qa_model.config.hidden_size, 2], task_name="question_answering")
            # transfer weights for head from full model
            head.feed_forward.feed_forward[0].load_state_dict(full_qa_model.qa_outputs.state_dict())
            del full_qa_model

        return head

    def forward(self, query_embed, passage_embed, embedding: torch.Tensor,
                query_ment_start: torch.Tensor, query_ment_end: torch.Tensor, **kwargs):
        """
        One forward pass through the prediction head model, starting with language model output on token level.
        """

        indicies = torch.arange(embedding.size(0))
        start_passage_idx = kwargs['seq_2_start_t'][0]
        # -1 exclude [SEP] token
        # Calculate query score
        query_ment_start_end_score = self.feed_forward(query_embed)
        query_ment_sstart_end_score = query_ment_start_end_score[indicies, query_ment_start, 0].unsqueeze(-1)
        query_ment_estart_end_score = query_ment_start_end_score[indicies, query_ment_end, 1].unsqueeze(-1)

        # Query (Start, End) score
        query_start_embed = query_embed[indicies, query_ment_start]
        query_end_embed = query_embed[indicies, query_ment_end]
        query_start_end_embed = torch.cat((query_start_embed, query_end_embed), dim=1)
        query_ment_start_concat_end_score = self.start_end_ff(query_start_end_embed)
        query_concat_scores = torch.cat((query_ment_sstart_end_score,
                                         query_ment_estart_end_score,
                                         query_ment_start_concat_end_score), dim=1)

        query_score = torch.sum(query_concat_scores, dim=1) / 3

        # Calculate passage score
        passage_ment_start_end_score = self.feed_forward(passage_embed)
        passage_ment_start_end_score_ext = passage_ment_start_end_score.unsqueeze(2).expand(-1, -1, passage_embed.size(1), -1)

        # Passage (Start, End) scores
        passage_embed_ext = passage_embed.unsqueeze(2).expand(-1, -1, passage_embed.size(1), -1)
        passage_start_end_embed = torch.cat((passage_embed_ext, passage_embed_ext.transpose(1, 2)), dim=3)
        passage_ment_start_concat_end_score = self.start_end_ff(passage_start_end_embed)
        # Calculating the average or passage mention scores
        passage_concat_scores = torch.cat((passage_ment_start_end_score_ext, passage_ment_start_concat_end_score), dim=3)
        passage_score = torch.sum(passage_concat_scores, dim=3) / 3

        # Again for query, this time from the concat embedding (query, passage)
        comb_query_start_embed = embedding[indicies, query_ment_start]
        comb_query_end_embed = embedding[indicies, query_ment_end]
        comb_query_start_end_embed = torch.cat((comb_query_start_embed, comb_query_end_embed), dim=1)
        query_pairwise_score = self.pairwize(comb_query_start_end_embed)

        # pairwise mention/passage, calculate score on concat embedding
        comb_pass_embed = embedding[indicies, start_passage_idx:]
        # Append passage with [CLS] token
        cls_toks = embedding[indicies, 0].unsqueeze(1)
        comb_pass_embed = torch.cat((cls_toks, comb_pass_embed), dim=1)

        comb_pass_start_embed_ext = comb_pass_embed.unsqueeze(2).expand(-1, -1, comb_pass_embed.size(1), -1)
        comb_pass_start_end_embed_concat = torch.cat((comb_pass_start_embed_ext, comb_pass_start_embed_ext.transpose(1, 2)), dim=3)
        pass_pairwise_score = self.pairwize(comb_pass_start_end_embed_concat)

        query_score = query_score.unsqueeze(-1).expand(-1, passage_score.size(1)).unsqueeze(-1).expand(-1, -1, passage_score.size(2))
        query_passage_score = passage_score + query_score

        query_pairwise_score = query_pairwise_score.expand(-1, pass_pairwise_score.size(1)).\
            unsqueeze(-1).expand(-1, -1, pass_pairwise_score.size(2))
        query_pass_pairwise_score = query_pairwise_score + pass_pairwise_score.squeeze(-1)

        lambda_ = 0.5
        final_score = torch.mul(query_passage_score, lambda_) + torch.mul(query_pass_pairwise_score, (1 - lambda_))

        return final_score

    def logits_to_loss(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Combine predictions and labels to a per sample loss.
        """
        # todo explain how we only use first answer for train
        # labels.shape =  [batch_size, n_max_answers, 2]. n_max_answers is by default 6 since this is the
        # most that occurs in the SQuAD dev set. The 2 in the final dimension corresponds to [start, end]
        start_position = labels[:, 0, 0]
        end_position = labels[:, 0, 1]

        start_passage_idx = kwargs['seq_2_start_t'][0]

        # +1 for [CLS] token which is concat below
        start_position = start_position - start_passage_idx + 1
        end_position = end_position - start_passage_idx + 1
        start_position[start_position < 0] = 0
        end_position[end_position < 0] = 0

        start_end_positions = (start_position * logits.size(1)) + end_position
        logits_flat = torch.flatten(logits, start_dim=1, end_dim=-1)

        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits_flat, start_end_positions)
        return loss

    def temperature_scale(self, logits: torch.Tensor):
        return torch.div(logits, self.temperature_for_confidence)

    def calibrate_conf(self, logits, label_all):
        """
        Learning a temperature parameter to apply temperature scaling to calibrate confidence scores
        """
        logits = torch.cat(logits, dim=0)

        # To handle no_answer labels correctly (-1,-1), we set their start_position to 0. The logit at index 0 also refers to no_answer
        # TODO some language models do not have the CLS token at position 0. For these models, we need to map start_position==-1 to the index of CLS token
        start_position = [label[0][0] if label[0][0] >= 0 else 0 for label in label_all]
        end_position = [label[0][1] if label[0][1] >= 0 else 0 for label in label_all]

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        start_position = torch.tensor(start_position)
        if len(start_position.size()) > 1:
            start_position = start_position.squeeze(-1)
        end_position = torch.tensor(end_position)
        if len(end_position.size()) > 1:
            end_position = end_position.squeeze(-1)

        ignored_index = start_logits.size(1) - 1
        start_position.clamp_(0, ignored_index)
        end_position.clamp_(0, ignored_index)

        nll_criterion = CrossEntropyLoss()

        optimizer = optim.LBFGS([self.temperature_for_confidence], lr=0.01, max_iter=50)

        def eval_start_end_logits():
            loss = nll_criterion(self.temperature_scale(start_logits), start_position.to(device=start_logits.device)) + \
                   nll_criterion(self.temperature_scale(end_logits), end_position.to(device=end_logits.device))
            loss.backward()
            return loss

        optimizer.step(eval_start_end_logits)

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

        # logits is of shape [batch_size, max_seq_len, max_seq_len].
        # Calculate a few useful variables
        batch_size = logits.size()[0]
        max_seq_len = logits.shape[1]  # target dim

        # disqualify answers where end < start
        # (set the lower triangular matrix to low value, excluding diagonal)
        indices = torch.tril_indices(max_seq_len, max_seq_len, offset=-1, device=logits.device)
        logits[:, indices[0][:], indices[1][:]] = -888

        # disqualify answers where answer span is greater than max_answer_length
        # (set the upper triangular matrix to low value, excluding diagonal)
        indices_long_span = torch.triu_indices(max_seq_len, max_seq_len, offset=max_answer_length,
                                               device=logits.device)
        logits[:, indices_long_span[0][:], indices_long_span[1][:]] = -777

        # disqualify answers where start=0, but end != 0
        logits[:, 0, 1:] = -666

        # Turn 1d span_mask vectors into 2d span_mask along 2 different axes
        # span mask has:
        #   0 for every position that is never a valid start or end index (question tokens, mid and end special tokens, padding)
        #   1 everywhere else
        idxs = torch.arange(logits.size(0))
        passage_span_mask = span_mask[idxs, seq_2_start_t[0]:]
        # add cls token
        cls_mask = span_mask[idxs, 0].unsqueeze(-1)
        passage_span_mask = torch.cat((cls_mask, passage_span_mask), dim=1)

        span_mask_start = passage_span_mask.unsqueeze(2).expand(-1, -1, max_seq_len)
        span_mask_end = passage_span_mask.unsqueeze(1).expand(-1, max_seq_len, -1)
        span_mask_2d = span_mask_start + span_mask_end
        # disqualify spans where either start or end is on an invalid token
        invalid_indices = torch.nonzero((span_mask_2d != 2), as_tuple=True)
        logits[invalid_indices[0][:], invalid_indices[1][:], invalid_indices[2][:]] = -999

        # Sort the candidate answers by their score. Sorting happens on the flattened matrix.
        # flat_sorted_indices.shape: (batch_size, max_seq_len^2, 1)
        flat_scores = logits.view(batch_size, -1)
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
                                                   logits[sample_idx],
                                                   sample_idx,
                                                   # start_matrix=start_matrix[sample_idx],
                                                   # end_matrix=end_matrix[sample_idx]
                                                   start_matrix=None,
                                                   end_matrix=None
                                                   )
            all_top_n.append(sample_top_n)

        return all_top_n

    def get_top_candidates(self, sorted_candidates, start_end_matrix, sample_idx: int, start_matrix, end_matrix):
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

        start_end_flat = torch.flatten(start_end_matrix)
        start_end_matrix_softmax = torch.softmax(start_end_flat, dim=0)
        start_end_matrix_softmax = start_end_matrix_softmax.view(start_end_matrix.size())
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
                if self.duplicate_filtering > -1 and (
                        start_idx in start_idx_candidates or end_idx in end_idx_candidates):
                    continue
                score = start_end_matrix[start_idx][end_idx].item()
                confidence = start_end_matrix_softmax[start_idx][end_idx].item()
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

        no_answer_score = start_end_matrix[0, 0].item()
        no_answer_confidence = start_end_matrix_softmax[0][0]
        top_candidates.append(QACandidate(offset_answer_start=0,
                                          offset_answer_end=0,
                                          score=no_answer_score,
                                          answer_type="no_answer",
                                          offset_unit="token",
                                          aggregation_level="passage",
                                          passage_id=None,
                                          confidence=no_answer_confidence))

        return top_candidates

    def formatted_preds(self, preds: List[QACandidate], baskets: List[SampleBasket],
                        logits: Optional[torch.Tensor] = None, **kwargs):
        """
        Takes a list of passage level predictions, each corresponding to one sample, and converts them into document level
        predictions. Leverages information in the SampleBaskets. Assumes that we are being passed predictions from
        ALL samples in the one SampleBasket i.e. all passages of a document. Logits should be None, because we have
        already converted the logits to predictions before calling formatted_preds.
        (see Inferencer._get_predictions_and_aggregate()).
        """
        # Unpack some useful variables
        # passage_start_t is the token index of the passage relative to the document (usually a multiple of doc_stride)
        # seq_2_start_t is the token index of the first token in passage relative to the input sequence (i.e. number of
        # special tokens and question tokens that come before the passage tokens)
        if logits or preds is None:
            logger.error("QuestionAnsweringHead.formatted_preds() expects preds as input and logits to be None \
                            but was passed something different")
        samples = [s for b in baskets for s in b.samples]  # type: ignore
        ids = [s.id for s in samples]
        passage_start_t = [s.features[0]["passage_start_t"] for s in samples]  # type: ignore
        seq_2_start_t = [s.features[0]["seq_2_start_t"] for s in samples]  # type: ignore

        # Aggregate passage level predictions to create document level predictions.
        # This method assumes that all passages of each document are contained in preds
        # i.e. that there are no incomplete documents. The output of this step
        # are prediction spans
        preds_d = self.aggregate_preds(preds, passage_start_t, ids, seq_2_start_t)

        # Separate top_preds list from the no_ans_gap float.
        top_preds, no_ans_gaps = zip(*preds_d)

        # Takes document level prediction spans and returns string predictions
        doc_preds = self.to_qa_preds(top_preds, no_ans_gaps, baskets)

        return doc_preds

    def to_qa_preds(self, top_preds, no_ans_gaps, baskets):
        """
        Groups Span objects together in a QAPred object
        """
        ret = []

        # Iterate over each set of document level prediction
        for pred_d, no_ans_gap, basket in zip(top_preds, no_ans_gaps, baskets):
            # Unpack document offsets, clear text and id
            token_offsets = basket.raw["document_offsets"]
            pred_id = basket.id_external if basket.id_external else basket.id_internal

            # These options reflect the different input dicts that can be assigned to the basket
            # before any kind of normalization or preprocessing can happen
            question_names = ["question_text", "qas", "questions"]
            doc_names = ["document_text", "context", "text"]

            document_text = try_get(doc_names, basket.raw)
            question = self.get_question(question_names, basket.raw)
            ground_truth = self.get_ground_truth(basket)

            curr_doc_pred = QAPred(id=pred_id,
                                   prediction=pred_d,
                                   context=document_text,
                                   question=question,
                                   token_offsets=token_offsets,
                                   context_window_size=self.context_window_size,
                                   aggregation_level="document",
                                   ground_truth_answer=ground_truth,
                                   no_answer_gap=no_ans_gap)

            ret.append(curr_doc_pred)
        return ret

    @staticmethod
    def get_ground_truth(basket: SampleBasket):
        if "answers" in basket.raw:
            return basket.raw["answers"]
        elif "annotations" in basket.raw:
            return basket.raw["annotations"]
        else:
            return None

    @staticmethod
    def get_question(question_names: List[str], raw_dict: Dict):
        # For NQ style dicts
        qa_name = None
        if "qas" in raw_dict:
            qa_name = "qas"
        elif "question" in raw_dict:
            qa_name = "question"
        if qa_name:
            if type(raw_dict[qa_name][0]) == dict:
                return raw_dict[qa_name][0]["question"]
        return try_get(question_names, raw_dict)

    def aggregate_preds(self, preds, passage_start_t, ids, seq_2_start_t=None, labels=None):
        """
        Aggregate passage level predictions to create document level predictions.
        This method assumes that all passages of each document are contained in preds
        i.e. that there are no incomplete documents. The output of this step
        are prediction spans. No answer is represented by a (-1, -1) span on the document level
        """
        # Initialize some variables
        n_samples = len(preds)
        all_basket_preds = {}
        all_basket_labels = {}

        # Iterate over the preds of each sample - remove final number which is the sample id and not needed for aggregation
        for sample_idx in range(n_samples):
            basket_id = ids[sample_idx]
            basket_id = basket_id.split("-")[:-1]
            basket_id = "-".join(basket_id)

            # curr_passage_start_t is the token offset of the current passage
            # It will always be a multiple of doc_stride
            # Need to remove the CLS token
            curr_passage_start_t = passage_start_t[sample_idx] - 1

            # This is to account for the fact that all model input sequences start with some special tokens
            # and also the question tokens before passage tokens.
            # if seq_2_start_t:
            #     cur_seq_2_start_t = seq_2_start_t[sample_idx]
            #     curr_passage_start_t -= cur_seq_2_start_t

            # Converts the passage level predictions+labels to document level predictions+labels. Note
            # that on the passage level a no answer is (0,0) but at document level it is (-1,-1) since (0,0)
            # would refer to the first token of the document
            pred_d = self.pred_to_doc_idxs(preds[sample_idx], curr_passage_start_t)
            if labels:
                label_d = self.label_to_doc_idxs(labels[sample_idx], curr_passage_start_t)

            # Initialize the basket_id as a key in the all_basket_preds and all_basket_labels dictionaries
            if basket_id not in all_basket_preds:
                all_basket_preds[basket_id] = []
                all_basket_labels[basket_id] = []

            # Add predictions and labels to dictionary grouped by their basket_ids
            all_basket_preds[basket_id].append(pred_d)
            if labels:
                all_basket_labels[basket_id].append(label_d)

        # Pick n-best predictions and remove repeated labels
        all_basket_preds = {k: self.reduce_preds(v) for k, v in all_basket_preds.items()}
        if labels:
            all_basket_labels = {k: self.reduce_labels(v) for k, v in all_basket_labels.items()}

        # Return aggregated predictions in order as a list of lists
        keys = [k for k in all_basket_preds]
        aggregated_preds = [all_basket_preds[k] for k in keys]
        if labels:
            labels = [all_basket_labels[k] for k in keys]
            return aggregated_preds, labels
        else:
            return aggregated_preds

    @staticmethod
    def reduce_labels(labels):
        """
        Removes repeat answers. Represents a no answer label as (-1,-1)
        """
        positive_answers = [(start, end) for x in labels for start, end in x if not (start == -1 and end == -1)]
        if not positive_answers:
            return [(-1, -1)]
        else:
            return list(set(positive_answers))

    def reduce_preds(self, preds):
        """
        This function contains the logic for choosing the best answers from each passage. In the end, it
        returns the n_best predictions on the document level.
        """
        # Initialize variables
        passage_no_answer = []
        passage_best_score = []
        passage_best_confidence = []
        no_answer_scores = []
        no_answer_confidences = []
        n_samples = len(preds)

        # Iterate over the top predictions for each sample
        for sample_idx, sample_preds in enumerate(preds):
            best_pred = sample_preds[0]
            best_pred_score = best_pred.score
            best_pred_confidence = best_pred.confidence
            no_answer_score, no_answer_confidence = self.get_no_answer_score_and_confidence(sample_preds)
            no_answer_score += self.no_ans_boost
            # TODO we might want to apply some kind of a no_ans_boost to no_answer_confidence too
            no_answer = no_answer_score > best_pred_score
            passage_no_answer.append(no_answer)
            no_answer_scores.append(no_answer_score)
            no_answer_confidences.append(no_answer_confidence)
            passage_best_score.append(best_pred_score)
            passage_best_confidence.append(best_pred_confidence)

        # Get all predictions in flattened list and sort by score
        pos_answers_flat = []
        for sample_idx, passage_preds in enumerate(preds):
            for qa_candidate in passage_preds:
                if not (qa_candidate.offset_answer_start == -1 and qa_candidate.offset_answer_end == -1):
                    pos_answers_flat.append(QACandidate(offset_answer_start=qa_candidate.offset_answer_start,
                                                        offset_answer_end=qa_candidate.offset_answer_end,
                                                        score=qa_candidate.score,
                                                        answer_type=qa_candidate.answer_type,
                                                        offset_unit="token",
                                                        aggregation_level="document",
                                                        passage_id=str(sample_idx),
                                                        n_passages_in_doc=n_samples,
                                                        confidence=qa_candidate.confidence)
                                            )

        # TODO add switch for more variation in answers, e.g. if varied_ans then never return overlapping answers
        pos_answer_dedup = self.deduplicate(pos_answers_flat)

        # This is how much no_ans_boost needs to change to turn a no_answer to a positive answer (or vice versa)
        no_ans_gap = -min([nas - pbs for nas, pbs in zip(no_answer_scores, passage_best_score)])
        no_ans_gap_confidence = -min([nas - pbs for nas, pbs in zip(no_answer_confidences, passage_best_confidence)])

        # "no answer" scores and positive answers scores are difficult to compare, because
        # + a positive answer score is related to a specific text qa_candidate
        # - a "no answer" score is related to all input texts
        # Thus we compute the "no answer" score relative to the best possible answer and adjust it by
        # the most significant difference between scores.
        # Most significant difference: change top prediction from "no answer" to answer (or vice versa)
        best_overall_positive_score = max(x.score for x in pos_answer_dedup)
        best_overall_positive_confidence = max(x.confidence for x in pos_answer_dedup)
        no_answer_pred = QACandidate(offset_answer_start=-1,
                                     offset_answer_end=-1,
                                     score=best_overall_positive_score - no_ans_gap,
                                     answer_type="no_answer",
                                     offset_unit="token",
                                     aggregation_level="document",
                                     passage_id=None,
                                     n_passages_in_doc=n_samples,
                                     confidence=best_overall_positive_confidence - no_ans_gap_confidence)

        # Add no answer to positive answers, sort the order and return the n_best
        n_preds = [no_answer_pred] + pos_answer_dedup
        n_preds_sorted = sorted(n_preds,
                                key=lambda x: x.confidence if self.use_confidence_scores_for_ranking else x.score,
                                reverse=True)
        n_preds_reduced = n_preds_sorted[:self.n_best]
        return n_preds_reduced, no_ans_gap

    @staticmethod
    def deduplicate(flat_pos_answers):
        # Remove duplicate spans that might be twice predicted in two different passages
        seen = {}
        for qa_answer in flat_pos_answers:
            if (qa_answer.offset_answer_start, qa_answer.offset_answer_end) not in seen:
                seen[(qa_answer.offset_answer_start, qa_answer.offset_answer_end)] = qa_answer
            else:
                seen_score = seen[(qa_answer.offset_answer_start, qa_answer.offset_answer_end)].score
                if qa_answer.score > seen_score:
                    seen[(qa_answer.offset_answer_start, qa_answer.offset_answer_end)] = qa_answer
        return list(seen.values())

    @staticmethod
    def get_no_answer_score_and_confidence(preds):
        for qa_answer in preds:
            start = qa_answer.offset_answer_start
            end = qa_answer.offset_answer_end
            score = qa_answer.score
            confidence = qa_answer.confidence
            if start == -1 and end == -1:
                return score, confidence
        raise Exception

    @staticmethod
    def pred_to_doc_idxs(pred, passage_start_t):
        """
        Converts the passage level predictions to document level predictions. Note that on the doc level we
        don't have special tokens or question tokens. This means that a no answer
        cannot be prepresented by a (0,0) qa_answer but will instead be represented by (-1, -1)
        """
        new_pred = []
        for qa_answer in pred:
            start = qa_answer.offset_answer_start
            end = qa_answer.offset_answer_end
            if start == 0:
                start = -1
            else:
                start += passage_start_t
                if start < 0:
                    logger.error("Start token index < 0 (document level)")
            if end == 0:
                end = -1
            else:
                end += passage_start_t
                if end < 0:
                    logger.error("End token index < 0 (document level)")
            qa_answer.to_doc_level(start, end)
            new_pred.append(qa_answer)
        return new_pred

    @staticmethod
    def label_to_doc_idxs(label, passage_start_t):
        """
        Converts the passage level labels to document level labels. Note that on the doc level we
        don't have special tokens or question tokens. This means that a no answer
        cannot be prepresented by a (0,0) span but will instead be represented by (-1, -1)
        """
        new_label = []
        for start, end in label:
            # If there is a valid label
            if start > 0 or end > 0:
                new_label.append((start + passage_start_t, end + passage_start_t))
            # If the label is a no answer, we represent this as a (-1, -1) span
            # since there is no CLS token on the document level
            if start == 0 and end == 0:
                new_label.append((-1, -1))
        return new_label

    def prepare_labels(self, labels, start_of_word, **kwargs):
        return labels
