import logging
from typing import List, Tuple

from haystack.modeling.data_handler.samples import SampleBasket, Sample
from src.data_obj import TrainExample, QueryFeat
from src.override_classes.retriever.search_processor import CoreSearchSimilarityProcessor, QUERY_SPAN_START, \
    QUERY_SPAN_END

logger = logging.getLogger(__name__)


class CoreSearchContextProcessor(CoreSearchSimilarityProcessor):
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
        add_special_tokens=None
    ):
        super(CoreSearchContextProcessor, self).__init__(
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
            label_list,
            add_special_tokens
        )

    def get_query_feat(self, query_obj: TrainExample) -> Tuple[QueryFeat, List[str]]:
        query_feat, query_tokenized = super().get_query_feat(query_obj)
        self.validate_query_toks(query_obj, query_feat, query_tokenized)

        return query_feat, query_tokenized

    def validate_query_toks(self, query_obj, query_feat, query_tokenized):
        if self.add_special_tokens:
            if QUERY_SPAN_START in self.query_tokenizer.added_tokens_encoder and QUERY_SPAN_END in self.query_tokenizer.added_tokens_encoder:
                assert query_feat.input_ids[query_feat.query_event_start] == self.query_tokenizer.added_tokens_encoder[
                    QUERY_SPAN_START]
                assert query_feat.input_ids[query_feat.query_event_end] == self.query_tokenizer.added_tokens_encoder[
                    QUERY_SPAN_END]
            elif QUERY_SPAN_START.lower() in self.query_tokenizer.added_tokens_encoder and QUERY_SPAN_END.lower() in self.query_tokenizer.added_tokens_encoder:
                assert query_feat.input_ids[query_feat.query_event_start] == self.query_tokenizer.added_tokens_encoder[
                    QUERY_SPAN_START.lower()]
                assert query_feat.input_ids[query_feat.query_event_end] == self.query_tokenizer.added_tokens_encoder[
                    QUERY_SPAN_END.lower()]
            else:
                raise AssertionError("add_spatial_token=True and no spatial tokens in added_tokens_encoder list!")
            # Assert that mention is equal to the tokenized mention (i.e., mention span is currect)
            if query_obj.mention:
                query_lower = "".join(query_obj.mention).lower()
                token_query_lower = "".join(
                    [s.strip('##') for s in
                     query_tokenized[query_feat.query_event_start + 1:query_feat.query_event_end]])
                if query_lower != token_query_lower:
                    print(f"WARNING:Query ({query_lower}) != tokenized query ({token_query_lower}), ID={query_obj.id}")
        else:
            assert query_feat.input_ids[
                   query_feat.query_event_start:query_feat.query_event_end + 1] == self.query_tokenizer.convert_tokens_to_ids(
                self.query_tokenizer.tokenize(" ".join(query_obj.mention)))
            # Assert that mention is equal to the tokenized mention (i.e., mention span is currect)
            if query_obj.mention:
                query_lower = "".join(query_obj.mention).lower()
                token_query_lower = "".join([
                    s.strip('##') for s in query_tokenized[query_feat.query_event_start:query_feat.query_event_end + 1]
                ]).lower()

                if query_lower != token_query_lower:
                    print(f"WARNING:Query ({query_lower}) != tokenized query ({token_query_lower})")

    def tokenize_query(self, query_obj: TrainExample, max_query_length_exclude: int):
        query_tokenized = list()
        query_event_start_ind = query_event_end_ind = 0
        if self.add_special_tokens and QUERY_SPAN_END not in query_obj.context and QUERY_SPAN_START not in query_obj.context:
            self.add_query_bound(query_obj)
        for index, word in enumerate(query_obj.context):
            query_tokenized.extend(self.query_tokenizer.tokenize(word))
            if self.add_special_tokens:
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
            # query_tokenized = query_tokenized[0:max_query_length_exclude]
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

        return query_event_start_ind, query_event_end_ind, query_tokenized, query_input_mask

    def _convert_queries(self, baskets: List[SampleBasket]):
        for basket in baskets:
            # extract query, positive context passages and titles, hard-negative passages and titles
            clear_text, tokenized, features, query_feat = self.get_basket_tokens_feats(basket)
            sample = Sample(id="",
                            clear_text=clear_text,
                            tokenized=tokenized,
                            features=features)  # type: ignore
            basket.samples = [sample]
        return baskets

    def get_basket_tokens_feats(self, basket: SampleBasket):
        clear_text = {}
        tokenized = {}
        features = [{}]  # type: ignore
        query_feat = None
        if "query" in basket.raw:
            query_params = basket.raw
            if isinstance(query_params["query"], dict):
                query_params = query_params["query"]

            assert "query_id" in query_params and "start_index" in query_params and "end_index" in query_params, "Invalid sample"
            try:
                query_obj = {
                    "id": query_params["query_id"],
                    "goldChain": None,
                    "mention": query_params["query_mention"],
                    "startIndex": query_params["start_index"],
                    "endIndex": query_params["end_index"],
                    "context": query_params["query"].split(" "),
                    "positive_examples": None,
                    "negative_examples": None,
                    "bm25_query": None,
                    "dummy": False,
                }
                clear_text, tokenized, features, query_feat = self.tokenize_and_add_features(query_obj)
            except Exception as e:
                features = None  # type: ignore
        return clear_text, tokenized, features, query_feat
