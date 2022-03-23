import logging
from typing import List, Dict

from tqdm import tqdm
from transformers import BertTokenizer

from src.data_obj import QueryFeat, PassageFeat, TrainExample, Passage, SearchFeat

QUERY_SPAN_START = "[QSPAN_START]"
QUERY_SPAN_END = "[QSPAN_END]"

logger = logging.getLogger("event-search")


class Tokenization(object):
    def __init__(self, query_tok_file=None, passage_tok_file=None,
                 query_tokenizer=None, passage_tokenizer=None,
                 max_query_size=64, max_passage_size=180, add_spatial_tokens=False):
        self.max_query_size = max_query_size
        self.max_passage_size = max_passage_size
        if query_tok_file and passage_tok_file:
            self.query_tokenizer = BertTokenizer.from_pretrained(query_tok_file)
            self.passage_tokenizer = BertTokenizer.from_pretrained(passage_tok_file)
        elif query_tokenizer and passage_tokenizer:
            self.query_tokenizer = query_tokenizer
            self.passage_tokenizer = passage_tokenizer
        else:
            raise IOError("No tokenizer initialization provided")

        if add_spatial_tokens and QUERY_SPAN_START.lower() not in self.query_tokenizer.added_tokens_encoder:
            self.query_tokenizer.add_tokens(QUERY_SPAN_START)
            self.query_tokenizer.add_tokens(QUERY_SPAN_END)

    def generate_train_search_feats(self,
                                    query_examples: List[TrainExample],
                                    passages_examples: List[Passage],
                                    negative_examples: int,
                                    add_qbound: bool = False) -> List[SearchFeat]:
        passages_examples_dict: Dict[str, Passage] = {passage.id: passage for passage in passages_examples}
        logger.info("Total examples loaded, queries=" + str(len(query_examples)) + ", passages=" + str(len(passages_examples_dict)))
        logger.info("Starting to generate examples...")
        passage_feats = dict()
        search_feats = list()
        total_gen_queries = 0
        for query_obj in tqdm(query_examples, "Loading Queries"):
            total_gen_queries += len(query_obj.positive_examples)
            ### REMOVE THIS!!!
            query_obj.context = query_obj.bm25_query.split(" ")
            ### END REMOVE THIS!!!
            query_feat = self.get_query_feat(query_obj, add_qbound)
            pos_passages = list()
            neg_passages = list()
            for pos_id in query_obj.positive_examples:
                if pos_id not in passage_feats:
                    passage_feats[pos_id] = self.get_passage_feat(passages_examples_dict[pos_id])
                pos_passages.append(passage_feats[pos_id])

            for neg_id in query_obj.negative_examples:
                if neg_id not in passage_feats:
                    passage_feats[neg_id] = self.get_passage_feat(passages_examples_dict[neg_id])
                # passage_cpy = copy.copy(passage_feats[neg_id])
                # passage_cpy.passage_event_start = passage_cpy.passage_event_end = 0
                neg_passages.append(passage_feats[neg_id])

            index = 0
            for pos_pass in pos_passages:
                if len(neg_passages) < index+negative_examples:
                    index = 0

                search_feats.append(SearchFeat(query_feat, pos_pass, neg_passages[index:index+negative_examples]))
                index += negative_examples

        print(f"Total generated queries = {total_gen_queries}")
        return search_feats

    def generate_query_feats(self, query_examples: List[TrainExample],
                             add_qbound: bool = False) -> List[QueryFeat]:
        query_feats = list()
        for query_obj in tqdm(query_examples, "Loading Queries"):
            query_feats.append(self.get_query_feat(query_obj, add_qbound))
        return query_feats

    def generate_passage_feats(self, passage_examples: List[Passage]) -> List[PassageFeat]:
        passage_feats = list()
        for passage_obj in tqdm(passage_examples, "Loading Passages"):
            passage_feats.append(self.get_passage_feat(passage_obj))
        return passage_feats

    def get_query_feat(self, query_obj: TrainExample, add_qbound: bool = False) -> QueryFeat:
        max_query_length_exclude = self.max_query_size - 2
        query_event_start, query_event_end, \
            query_tokenized, query_input_mask = self.tokenize_query(query_obj, max_query_length_exclude, add_qbound)

        query_tokenized.insert(0, "[CLS]")
        query_input_mask.insert(0, 1)
        query_event_start += 1
        query_event_end += 1

        query_tokenized.append("[SEP]")
        query_input_mask.append(0)
        query_segment_ids = [1] * len(query_tokenized)
        query_input_ids = self.query_tokenizer.convert_tokens_to_ids(query_tokenized)

        assert len(query_input_ids) == self.max_query_size
        assert len(query_input_mask) == self.max_query_size
        assert len(query_segment_ids) == self.max_query_size
        # if add_qbound:
        #     assert query_input_ids[query_event_start] == self.query_tokenizer.added_tokens_encoder[QUERY_SPAN_START]
        #     assert query_input_ids[query_event_end] == self.query_tokenizer.added_tokens_encoder[QUERY_SPAN_END]
        #     # Assert that mention is equal to the tokenized mention (i.e., mention span is currect)
        #     if query_obj.mention:
        #         query_lower = "".join(query_obj.mention).lower()
        #         token_query_lower = "".join(
        #             [s.strip('##') for s in query_tokenized[query_event_start+1:query_event_end]])
        #         if query_lower != token_query_lower:
        #             print(f"WARNING:Query ({query_lower}) != tokenized query ({token_query_lower})")
        # else:
        #     assert query_input_ids[query_event_start] == self.query_tokenizer.convert_tokens_to_ids(
        #         self.query_tokenizer.tokenize(query_obj.mention[0])[0])
        #     assert query_input_ids[query_event_end] == self.query_tokenizer.convert_tokens_to_ids(
        #         self.query_tokenizer.tokenize(query_obj.mention[-1])[-1])
        #     # Assert that mention is equal to the tokenized mention (i.e., mention span is currect)
        #     if query_obj.mention:
        #         query_lower = "".join(query_obj.mention).lower()
        #         token_query_lower = "".join([s.strip('##') for s in query_tokenized[query_event_start:query_event_end + 1]]).lower()
        #         if query_lower != token_query_lower:
        #             print(f"WARNING:Query ({query_lower}) != tokenized query ({token_query_lower})")

        return QueryFeat(query_input_ids=query_input_ids,
                         query_input_mask=query_input_mask,
                         query_segment_ids=query_segment_ids,
                         query_ref=query_obj,
                         query_event_start=query_event_start,
                         query_event_end=query_event_end)

    def get_passage_feat(self, passage_obj: Passage) -> PassageFeat:
        max_pass_length_exclude = self.max_passage_size - 2
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

        assert len(input_ids) == self.max_passage_size
        assert len(passage_input_mask) == self.max_passage_size
        assert len(passage_segment_ids) == self.max_passage_size

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

    def tokenize_query(self, query_obj: TrainExample, max_query_length_exclude: int, add_qbound: bool):
        query_tokenized = list()
        query_event_start_ind = query_event_end_ind = 0
        if add_qbound and QUERY_SPAN_END not in query_obj.context and QUERY_SPAN_START not in query_obj.context:
            self.add_query_bound(query_obj)
        for index, word in enumerate(query_obj.context):
            query_tokenized.extend(self.query_tokenizer.tokenize(word))
            if add_qbound:
                if word == QUERY_SPAN_START:
                    query_event_start_ind = len(query_tokenized) - 1
                elif word == QUERY_SPAN_END:
                    query_event_end_ind = len(query_tokenized) - 1
            else:
                if index == query_obj.startIndex:
                    query_event_start_ind = len(query_tokenized) - len(self.query_tokenizer.tokenize(word))
                if index == query_obj.endIndex:
                    query_event_end_ind = len(query_tokenized) - 1

        pointer_start = query_event_start_ind
        pointer_end = query_event_end_ind

        if len(query_tokenized) > max_query_length_exclude:
            query_tokenized = query_tokenized[0:max_query_length_exclude]
            # trimmed_query_tok = query_tokenized[pointer_start:pointer_end + 1]
            # while len(trimmed_query_tok) < max_query_length_exclude - 1:
            #     if pointer_end < len(query_tokenized) - 1:
            #         pointer_end += 1
            #         trimmed_query_tok.append(query_tokenized[pointer_end])
            #     if pointer_start > 0:
            #         pointer_start -= 1
            #         trimmed_query_tok.insert(0, query_tokenized[pointer_start])

            # query_tokenized = trimmed_query_tok
            # query_event_start_ind -= pointer_start
            # query_event_end_ind -= pointer_start

        query_input_mask = [1] * len(query_tokenized)
        while len(query_tokenized) < max_query_length_exclude:
            query_tokenized.append('[PAD]')
            query_input_mask.append(0)

        return query_event_start_ind, query_event_end_ind, query_tokenized, query_input_mask

    @staticmethod
    def add_query_bound(query_obj: TrainExample):
        query_obj.context.insert(query_obj.endIndex + 1, QUERY_SPAN_END)
        query_obj.context.insert(query_obj.startIndex, QUERY_SPAN_START)
