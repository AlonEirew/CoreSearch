import copy
import logging
import random
from typing import List, Dict

from tqdm import tqdm
from transformers import BertTokenizer

from src.data_obj import QueryFeat, PassageFeat, TrainExample, Passage, SearchFeat
from src.utils import io_utils

QUERY_SPAN_START = "[QSPAN_START]"
QUERY_SPAN_END = "[QSPAN_END]"

logger = logging.getLogger("event-search")


class Tokenization(object):
    def __init__(self, model_file=None, tokenizer=None):
        if model_file:
            self.tokenizer = BertTokenizer.from_pretrained(model_file)
        elif tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.tokenizer.add_tokens(QUERY_SPAN_START)
            self.tokenizer.add_tokens(QUERY_SPAN_END)

    def generate_queries_feats(self,
                               query_file: str,
                               passages_file: str,
                               max_query_length: int,
                               max_passage_length: int,
                               negative_sample_size: int,
                               remove_qbound: bool = False) -> List[SearchFeat]:
        query_examples: List[TrainExample] = io_utils.read_train_example_file(query_file)
        passages_examples: List[Passage] = io_utils.read_passages_file(passages_file)
        passages_examples: Dict[str, Passage] = {passage.id: passage for passage in passages_examples}
        logger.info("Done loading examples file, queries-" + query_file + ", passages-" + passages_file)
        logger.info(
            "Total examples loaded, queries=" + str(len(query_examples)) + ", passages=" + str(len(passages_examples)))
        logger.info("Starting to generate examples...")
        query_feats = dict()
        passage_feats = dict()
        search_feats = list()
        for query_obj in tqdm(query_examples, "Loading Queries"):
            query_feat = self.get_query_feat(query_obj, max_query_length, remove_qbound)
            query_feats[query_obj.id] = query_feat
            pos_passages = list()
            neg_passages = list()
            for pos_id in query_obj.positive_examples:
                if pos_id not in passage_feats:
                    passage_feats[pos_id] = self.get_passage_feat(passages_examples[pos_id], max_passage_length)
                pos_passages.append(passage_feats[pos_id])

            for neg_id in query_obj.negative_examples:
                if neg_id not in passage_feats:
                    passage_feats[neg_id] = self.get_passage_feat(passages_examples[neg_id], max_passage_length)
                passage_cpy = copy.copy(passage_feats[neg_id])
                passage_cpy.passage_event_start = passage_cpy.passage_event_end = 0
                neg_passages.append(passage_cpy)

            for pos_pass in pos_passages:
                if len(neg_passages) > negative_sample_size:
                    search_feats.append(
                        SearchFeat(query_feat, pos_pass, random.sample(neg_passages, negative_sample_size)))
                else:
                    search_feats.append(
                        SearchFeat(query_feat, pos_pass, random.choices(neg_passages, k=negative_sample_size)))

        return search_feats

    def get_query_feat(self, query_obj: TrainExample, max_query_length: int, remove_qbound: bool = False) -> QueryFeat:
        max_query_length_exclude = max_query_length - 1
        query_event_start, query_event_end, \
        query_tokenized, query_input_mask = self.query_tokenization(query_obj, max_query_length_exclude, remove_qbound)

        query_tokenized.append("[SEP]")
        query_input_mask.append(0)
        query_segment_ids = [1] * len(query_tokenized)

        # if query_event_start != 0 and query_event_end != 0:
        #     query_event_start += max_passage_length
        #     query_event_end += max_passage_length

        query_input_ids = self.tokenizer.convert_tokens_to_ids(query_tokenized)

        assert len(query_input_ids) == max_query_length
        assert len(query_input_mask) == max_query_length
        assert len(query_segment_ids) == max_query_length

        return QueryFeat(query_input_ids=query_input_ids,
                         query_input_mask=query_input_mask,
                         query_segment_ids=query_segment_ids,
                         query_id=query_obj.id,
                         query_event_start=query_event_start,
                         query_event_end=query_event_end)

    def get_passage_feat(self, passage_obj: Passage, max_passage_length: int) -> PassageFeat:
        max_pass_length_exclude = max_passage_length - 2
        passage_event_start, passage_event_end, passage_end_bound, passage_tokenized, passage_input_mask = \
            self.passage_tokenization(passage_obj, max_pass_length_exclude)
        passage_tokenized.insert(0, "[CLS]")
        passage_input_mask.insert(0, 1)
        if passage_event_start != 0 and passage_event_end != 0:
            passage_event_start += 1
            passage_event_end += 1

        passage_tokenized.append("[SEP]")
        passage_input_mask.append(0)
        passage_segment_ids = [0] * len(passage_tokenized)

        input_ids = self.tokenizer.convert_tokens_to_ids(passage_tokenized)

        assert len(input_ids) == max_passage_length
        assert len(passage_input_mask) == max_passage_length
        assert len(passage_segment_ids) == max_passage_length

        return PassageFeat(
            passage_input_ids=input_ids,
            passage_input_mask=passage_input_mask,
            passage_segment_ids=passage_segment_ids,
            passage_id=passage_obj.id,
            passage_event_start=passage_event_start,
            passage_event_end=passage_event_end,
            passage_end_bound=passage_end_bound
        )

    def passage_tokenization(self, passage_obj: Passage, max_pass_length_exclude: int):
        passage_tokenized = list()
        start_index = passage_obj.startIndex
        end_index = passage_obj.endIndex
        pass_context = passage_obj.context
        passage_event_start_ind = passage_event_end_ind = 0
        for i in range(len(pass_context)):
            word_tokens = self.tokenizer.tokenize(pass_context[i])
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

    def query_tokenization(self, query_obj: TrainExample, max_query_length_exclude: int, remove_qbound: bool):
        query_tokenized = list()
        query_event_start_ind = query_event_end_ind = 0
        self.add_query_bound(query_obj)
        for word in query_obj.context:
            query_tokenized.extend(self.tokenizer.tokenize(word))
            if word == QUERY_SPAN_START:
                query_event_start_ind = len(query_tokenized)
            elif word == QUERY_SPAN_END:
                query_event_end_ind = len(query_tokenized) - 2

        pointer_start = query_event_start_ind - 1
        pointer_end = query_event_end_ind + 1
        if remove_qbound:
            query_tokenized.remove(QUERY_SPAN_START)
            query_tokenized.remove(QUERY_SPAN_END)
            query_event_start_ind -= 1
            query_event_end_ind -= 1
            pointer_start = query_event_start_ind
            pointer_end = query_event_end_ind

        if len(query_tokenized) > max_query_length_exclude:
            trimmed_query_tok = query_tokenized[pointer_start:pointer_end + 1]
            while len(trimmed_query_tok) < max_query_length_exclude - 1:
                if pointer_end < len(query_tokenized) - 1:
                    pointer_end += 1
                    trimmed_query_tok.append(query_tokenized[pointer_end])
                if pointer_start > 0:
                    pointer_start -= 1
                    trimmed_query_tok.insert(0, query_tokenized[pointer_start])

            query_tokenized = trimmed_query_tok
            query_event_start_ind -= pointer_start
            query_event_end_ind -= pointer_start

        query_input_mask = [1] * len(query_tokenized)
        while len(query_tokenized) < max_query_length_exclude:
            query_tokenized.append('[PAD]')
            query_input_mask.append(0)

        assert "".join(query_obj.mention) == "".join(
            [s.strip('##') for s in query_tokenized[query_event_start_ind:query_event_end_ind + 1]])
        return query_event_start_ind, query_event_end_ind, query_tokenized, query_input_mask

    @staticmethod
    def add_query_bound(query_obj: TrainExample):
        query_obj.context.insert(query_obj.endIndex + 1, QUERY_SPAN_END)
        query_obj.context.insert(query_obj.startIndex, QUERY_SPAN_START)
