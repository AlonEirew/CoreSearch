from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
from transformers.file_utils import ModelOutput


class Feat(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 feat_id):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.feat_id = feat_id


class QueryFeat(Feat):
    def __init__(self,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 query_id,
                 query_event_start,
                 query_event_end):

        super().__init__(query_input_ids, query_input_mask, query_segment_ids, query_id)
        self.query_event_start = query_event_start
        self.query_event_end = query_event_end


class PassageFeat(Feat):
    def __init__(self,
                 passage_input_ids,
                 passage_input_mask,
                 passage_segment_ids,
                 passage_id,
                 passage_event_start=None,
                 passage_event_end=None,
                 passage_end_bound=None,
                 is_positive=None):

        super().__init__(passage_input_ids, passage_input_mask, passage_segment_ids, passage_id)
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
        self.context = json_obj["context"]
        if "dummy" in json_obj and json_obj["dummy"]:
            self.id = None
            self.goldChain = None
            self.mention = None
            self.startIndex = None
            self.endIndex = None
            return

        self.id = json_obj["id"]
        self.goldChain = str(json_obj["goldChain"])
        self.mention = json_obj["mention"]
        self.startIndex = json_obj["startIndex"]
        self.endIndex = json_obj["endIndex"]
        self.context = json_obj["context"]

    @staticmethod
    def list_to_map(list_to_convert) -> Dict:
        assert list_to_convert
        return {obj.id: obj for obj in list_to_convert}


class Passage(BasicMent):
    def __init__(self, json_obj: Dict):
        super().__init__(json_obj)
        self.score = 0.0
        self.answer = None
        self.offsets_in_document = None
        if "score" in json_obj:
            self.score = json_obj["score"]
        if "answer" in json_obj:
            self.answer = json_obj["answer"]
        if "offsets_in_document" in json_obj:
            self.offsets_in_document = json_obj["offsets_in_document"]


class Query(BasicMent):
    def __init__(self, json_obj: Dict):
        super().__init__(json_obj)


class Cluster(object):
    def __init__(self, cluster_obj: Dict):
        self.cluster_id = str(cluster_obj["clusterId"])
        self.cluster_title = cluster_obj["clusterTitle"]
        self.mention_ids = [str(id_) for id_ in cluster_obj["mentionIds"]]


class QueryResult(object):
    def __init__(self, query: Query, results: List[Passage], searched_query: str = None):
        self.query = query
        self.results = results
        self.searched_query = searched_query


class TrainExample(BasicMent):
    def __init__(self, json_obj: Dict):
        super().__init__(json_obj)
        if "dummy" in json_obj and json_obj["dummy"]:
            return

        self.positive_examples = json_obj["positive_examples"]
        self.negative_examples = json_obj["negative_examples"]
        self.bm25_query = json_obj["bm25_query"]
        self.answers = set()


class DPRContext(object):
    def __init__(self,
                 title: str,
                 text: str,
                 score: int,
                 title_score: int,
                 passage_id: str):
        self.title = title
        self.text = text
        self.score = score
        self.title_score = title_score
        self.passage_id = passage_id


class DPRExample(object):
    def __init__(self,
                 dataset: str,
                 question: str,
                 answers: List[str],
                 positive_ctxs: List[DPRContext],
                 negative_ctxs: List[DPRContext],
                 hard_negative_ctxs: List[DPRContext]):

        self.dataset = dataset
        self.question = question
        self.answers = answers
        self.positive_ctxs = positive_ctxs
        self.negative_ctxs = negative_ctxs
        self.hard_negative_ctxs = hard_negative_ctxs
