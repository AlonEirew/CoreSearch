import logging
from typing import List, Dict

from tqdm import tqdm
from transformers import BertTokenizer

from src.data_obj import TrainExample, Passage, SearchFeat

# QUERY_SPAN_START = "[QSPAN_START]"
# QUERY_SPAN_END = "[QSPAN_END]"

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
