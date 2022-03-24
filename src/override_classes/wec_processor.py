import json
import logging
import os
import random
import uuid
from pathlib import Path
from typing import List, Optional, Any, Tuple

from haystack.modeling.data_handler.processor import TextSimilarityProcessor, _download_extract_downstream_data
from haystack.modeling.data_handler.samples import SampleBasket, Sample
from tqdm import tqdm

from src.data_obj import TrainExample, QueryFeat, Passage, PassageFeat

logger = logging.getLogger(__name__)
QUERY_SPAN_START = "[QSPAN_START]"
QUERY_SPAN_END = "[QSPAN_END]"


class WECSimilarityProcessor(TextSimilarityProcessor):
    def __init__(
        self,
        query_tokenizer,
        passage_tokenizer,
        max_seq_len_query,
        max_seq_len_passage,
        data_dir="",
        metric=None,
        train_filename="train.json",
        dev_filename=None,
        test_filename="test.json",
        dev_split=0.1,
        proxies=None,
        max_samples=None,
        embed_title=True,
        num_positives=1,
        num_hard_negatives=1,
        shuffle_negatives=True,
        shuffle_positives=False,
        label_list=None,
        add_spatial_tokens=False
    ):
        super(WECSimilarityProcessor, self).__init__(
            query_tokenizer,
            passage_tokenizer,
            max_seq_len_query,
            max_seq_len_passage,
            data_dir,
            metric,
            train_filename,
            dev_filename,
            test_filename,
            dev_split,
            proxies,
            max_samples,
            embed_title,
            num_positives,
            num_hard_negatives,
            shuffle_negatives,
            shuffle_positives,
            label_list
        )

        self.add_spatial_tokens = add_spatial_tokens

        if self.add_spatial_tokens and QUERY_SPAN_START.lower() not in self.query_tokenizer.added_tokens_encoder:
            self.query_tokenizer.add_tokens(QUERY_SPAN_START)
            self.query_tokenizer.add_tokens(QUERY_SPAN_END)

    def generate_query_feats(self, query_examples: List[TrainExample]) -> List[QueryFeat]:
        query_feats = list()
        for query_obj in tqdm(query_examples, "Loading Queries"):
            query_feat, _ = self.get_query_feat(query_obj)
            query_feats.append(query_feat)
        return query_feats

    def generate_passage_feats(self, passage_examples: List[Passage]) -> List[PassageFeat]:
        passage_feats = list()
        for passage_obj in tqdm(passage_examples, "Loading Passages"):
            passage_feats.append(self.get_passage_feat(passage_obj))
        return passage_feats

    def get_query_feat(self, query_obj: TrainExample) -> Tuple[QueryFeat, List[str]]:
        max_query_length_exclude = self.max_seq_len_query - 2
        query_event_start, query_event_end, \
            query_tokenized, query_input_mask = self.tokenize_query(query_obj, max_query_length_exclude)

        query_tokenized.insert(0, "[CLS]")
        query_input_mask.insert(0, 1)
        query_event_start += 1
        query_event_end += 1

        query_tokenized.append("[SEP]")
        query_input_mask.append(0)
        query_segment_ids = [1] * len(query_tokenized)
        query_input_ids = self.query_tokenizer.convert_tokens_to_ids(query_tokenized)

        assert len(query_input_ids) == self.max_seq_len_query
        assert len(query_input_mask) == self.max_seq_len_query
        assert len(query_segment_ids) == self.max_seq_len_query

        query_feat = QueryFeat(query_input_ids=query_input_ids, query_input_mask=query_input_mask,
                               query_segment_ids=query_segment_ids, query_ref=query_obj,
                               query_event_start=query_event_start,
                               query_event_end=query_event_end)
        return query_feat, query_tokenized

    def get_passage_feat(self, passage_obj: Passage) -> PassageFeat:
        max_pass_length_exclude = self.max_seq_len_passage - 2
        passage_event_start, passage_event_end, passage_end_bound, passage_tokenized, passage_input_mask = \
            self.tokenize_passage(passage_obj, max_pass_length_exclude)
        passage_tokenized.insert(0, "[CLS]")
        passage_input_mask.insert(0, 1)
        if passage_event_start != 0 and passage_event_end != 0:
            passage_event_start += 1
            passage_event_end += 1

        passage_tokenized.append("[SEP]")
        passage_input_mask.append(0)
        passage_segment_ids = [0] * len(passage_tokenized)

        input_ids = self.passage_tokenizer.convert_tokens_to_ids(passage_tokenized)

        assert len(input_ids) == self.max_seq_len_passage
        assert len(passage_input_mask) == self.max_seq_len_passage
        assert len(passage_segment_ids) == self.max_seq_len_passage

        return PassageFeat(
            passage_input_ids=input_ids,
            passage_input_mask=passage_input_mask,
            passage_segment_ids=passage_segment_ids,
            passage_ref=passage_obj,
            passage_event_start=passage_event_start,
            passage_event_end=passage_event_end,
            passage_end_bound=passage_end_bound
        )

    def tokenize_passage(self, passage_obj: Passage, max_pass_length_exclude: int):
        passage_tokenized = list()
        start_index = passage_obj.startIndex
        end_index = passage_obj.endIndex
        pass_context = passage_obj.context
        passage_event_start_ind = passage_event_end_ind = 0
        for i in range(len(pass_context)):
            word_tokens = self.passage_tokenizer.tokenize(pass_context[i])
            passage_tokenized.extend(word_tokens)
            if i == start_index:
                passage_event_start_ind = len(passage_tokenized) - len(word_tokens)
            if i == end_index:
                passage_event_end_ind = len(passage_tokenized) - 1

        passage_input_mask = [1] * len(passage_tokenized)
        if len(passage_tokenized) > max_pass_length_exclude:
            passage_tokenized = passage_tokenized[:max_pass_length_exclude]
            passage_input_mask = passage_input_mask[:max_pass_length_exclude]
            passage_end_bound = max_pass_length_exclude
        else:
            passage_end_bound = len(passage_tokenized)
            while len(passage_tokenized) < max_pass_length_exclude:
                passage_tokenized.append('[PAD]')
                passage_input_mask.append(0)

        if passage_event_end_ind > len(passage_tokenized):
            passage_event_start_ind = 0
            passage_event_end_ind = 0

        assert passage_event_start_ind <= passage_event_end_ind
        return passage_event_start_ind, passage_event_end_ind, passage_end_bound, passage_tokenized, passage_input_mask

    @staticmethod
    def add_query_bound(query_obj: TrainExample):
        query_obj.context.insert(query_obj.endIndex + 1, QUERY_SPAN_END)
        query_obj.context.insert(query_obj.startIndex, QUERY_SPAN_START)

    def file_to_dicts(self, file: str) -> List[dict]:
        dicts = self._read_dpr_json(file, max_samples=self.max_samples, num_hard_negatives=self.num_hard_negatives,
                                    num_positives=self.num_positives, shuffle_negatives=self.shuffle_negatives,
                                    shuffle_positives=self.shuffle_positives)

        # shuffle dicts to make sure that similar positive passages do not end up in one batch
        dicts = random.sample(dicts, len(dicts))
        return dicts

    @staticmethod
    def _read_dpr_json(file: str, max_samples: Optional[int] = None, proxies: Any = None, num_hard_negatives: int = 1,
                       num_positives: int = 1, shuffle_negatives: bool = True, shuffle_positives: bool = False):

        # get remote dataset if needed
        if not (os.path.exists(file)):
            logger.info(f" Couldn't find {file} locally. Trying to download ...")
            _download_extract_downstream_data(file, proxies=proxies)

        if Path(file).suffix.lower() == ".jsonl":
            dicts = []
            with open(file, encoding='utf-8') as f:
                for line in f:
                    dicts.append(json.loads(line))
        else:
            dicts = json.load(open(file, encoding='utf-8'))

        if max_samples:
            dicts = random.sample(dicts, min(max_samples, len(dicts)))

        # convert DPR dictionary to standard dictionary
        query_json_keys = ["question", "questions", "query"]
        positive_context_json_keys = ["positive_contexts", "positive_ctxs", "positive_context", "positive_ctx"]
        hard_negative_json_keys = ["hard_negative_contexts", "hard_negative_ctxs", "hard_negative_context",
                                   "hard_negative_ctx"]
        standard_dicts = []
        for dict in dicts:
            sample = {}
            passages = []
            for key, val in dict.items():
                if key in query_json_keys:
                    sample["query"] = val
                if key == "query_id":
                    sample["query_id"] = val
                if key == "start_index":
                    sample["start_index"] = val
                if key == "end_index":
                    sample["end_index"] = val
                if key == "query_mention":
                    sample["query_mention"] = val
                elif key in positive_context_json_keys:
                    if shuffle_positives:
                        random.shuffle(val)
                    for passage in val[:num_positives]:
                        passages.append({
                            "title": passage["title"],
                            "text": passage["text"],
                            "label": "positive",
                            "external_id": passage.get("passage_id", uuid.uuid4().hex.upper()[0:8])
                        })
                elif key in hard_negative_json_keys:
                    if shuffle_negatives:
                        random.shuffle(val)
                    for passage in val[:num_hard_negatives]:
                        passages.append({
                            "title": passage["title"],
                            "text": passage["text"],
                            "label": "hard_negative",
                            "external_id": passage.get("passage_id", uuid.uuid4().hex.upper()[0:8])
                        })
            sample["passages"] = passages
            standard_dicts.append(sample)
        return standard_dicts

    def tokenize_query(self, query_obj, max_query_length_exclude):
        raise NotImplementedError("Method should be called from one of the subclasses")
