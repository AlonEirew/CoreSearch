from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
from transformers.file_utils import ModelOutput


class QueryFeat(object):
    def __init__(self,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 query_id,
                 query_event_start,
                 query_event_end,
                 ):

        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids
        self.query_id = query_id
        self.query_event_start = query_event_start
        self.query_event_end = query_event_end


class PassageFeat(object):
    def __init__(self,
                 passage_input_ids,
                 passage_input_mask,
                 passage_segment_ids,
                 passage_id,
                 passage_event_start=None,
                 passage_event_end=None,
                 passage_end_bound=None,
                 is_positive=None
                 ):

        self.passage_input_ids = passage_input_ids
        self.passage_input_mask = passage_input_mask
        self.passage_segment_ids = passage_segment_ids
        self.passage_id = passage_id
        self.passage_event_start = passage_event_start
        self.passage_event_end = passage_event_end
        self.passage_end_bound = passage_end_bound
        self.is_positive = is_positive


class SearchFeat(object):
    def __init__(self, query: QueryFeat, pos_passage: PassageFeat,
                 negative_passage: List[PassageFeat]):
        self.query = query
        self.positive_passage = pos_passage
        self.negative_passages = negative_passage


class SpanPredictFeat(object):
    def __init__(self, query_feat: QueryFeat, passage_feat: PassageFeat):
        self.query_feat = query_feat
        self.passage_feat = passage_feat


@dataclass
class QuestionAnsweringModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    query_hidden_states: torch.FloatTensor = None
    passage_hidden_states: torch.FloatTensor = None


class EvaluationObject(object):
    def __init__(self,
                 start_label,
                 end_label,
                 start_pred,
                 end_pred,
                 passage_bound,
                 query_event_start,
                 query_event_end):
        self.start_label = start_label
        self.end_label = end_label
        self.start_pred = start_pred
        self.end_pred = end_pred
        self.passage_bound = passage_bound
        self.query_event_start = query_event_start
        self.query_event_end = query_event_end


class BasicMent(object):
    def __init__(self, json_obj: Dict):
        self.id = json_obj["id"]
        self.cluster = json_obj["goldChain"]
        self.context = json_obj["context"]
        self.mention = json_obj["mention"]
        self.start_idx = json_obj["startIndex"]
        self.end_idx = json_obj["endIndex"]


class Passage(BasicMent):
    def __init__(self, json_obj: Dict):
        super().__init__(json_obj)
        if "score" in json_obj:
            self.score = json_obj["score"]


class Query(BasicMent):
    def __init__(self, json_obj: Dict):
        super().__init__(json_obj)


class Cluster(object):
    def __init__(self, cluster_obj: Dict):
        self.cluster_id = str(cluster_obj["clusterId"])
        self.mention_ids = [str(id_) for id_ in cluster_obj["mentionIds"]]


class QueryResult(object):
    def __init__(self, query: Query, results: List[Passage]):
        self.query = query
        self.results = results
