import logging
from typing import List, Tuple

from haystack.modeling.data_handler.samples import SampleBasket, Sample

from src.data_obj import TrainExample, QueryFeat
from src.override_classes.wec_processor import WECSimilarityProcessor, QUERY_SPAN_END, QUERY_SPAN_START

logger = logging.getLogger(__name__)


class WECContextProcessor(WECSimilarityProcessor):
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
        add_spatial_tokens=None
    ):
        super(WECContextProcessor, self).__init__(
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
            add_spatial_tokens
        )

    def get_query_feat(self, query_obj: TrainExample) -> Tuple[QueryFeat, List[str]]:
        query_feat, query_tokenized = super().get_query_feat(query_obj)
        if self.add_spatial_tokens:
            if QUERY_SPAN_START in self.query_tokenizer.added_tokens_encoder and QUERY_SPAN_END in self.query_tokenizer.added_tokens_encoder:
                assert query_feat.input_ids[query_feat.query_event_start] == self.query_tokenizer.added_tokens_encoder[QUERY_SPAN_START]
                assert query_feat.input_ids[query_feat.query_event_end] == self.query_tokenizer.added_tokens_encoder[QUERY_SPAN_END]
            elif QUERY_SPAN_START.lower() in self.query_tokenizer.added_tokens_encoder and QUERY_SPAN_END.lower() in self.query_tokenizer.added_tokens_encoder:
                assert query_feat.input_ids[query_feat.query_event_start] == self.query_tokenizer.added_tokens_encoder[QUERY_SPAN_START.lower()]
                assert query_feat.input_ids[query_feat.query_event_end] == self.query_tokenizer.added_tokens_encoder[QUERY_SPAN_END.lower()]
            else:
                raise AssertionError("add_spatial_token=True and no spatial tokens in added_tokens_encoder list!")
            # Assert that mention is equal to the tokenized mention (i.e., mention span is currect)
            if query_obj.mention:
                query_lower = "".join(query_obj.mention).lower()
                token_query_lower = "".join(
                    [s.strip('##') for s in query_tokenized[query_feat.query_event_start+1:query_feat.query_event_end]])
                if query_lower != token_query_lower:
                    print(f"WARNING:Query ({query_lower}) != tokenized query ({token_query_lower})")
        else:
            assert query_feat.input_ids[query_feat.query_event_start:query_feat.query_event_end + 1] == self.query_tokenizer.convert_tokens_to_ids(
                self.query_tokenizer.tokenize(" ".join(query_obj.mention)))
            # Assert that mention is equal to the tokenized mention (i.e., mention span is currect)
            if query_obj.mention:
                query_lower = "".join(query_obj.mention).lower()
                token_query_lower = "".join([
                    s.strip('##') for s in query_tokenized[query_feat.query_event_start:query_feat.query_event_end + 1]
                ]).lower()

                if query_lower != token_query_lower:
                    print(f"WARNING:Query ({query_lower}) != tokenized query ({token_query_lower})")

        return query_feat, query_tokenized

    def tokenize_query(self, query_obj: TrainExample, max_query_length_exclude: int):
        query_tokenized = list()
        query_event_start_ind = query_event_end_ind = 0
        if self.add_spatial_tokens and QUERY_SPAN_END not in query_obj.context and QUERY_SPAN_START not in query_obj.context:
            self.add_query_bound(query_obj)
        for index, word in enumerate(query_obj.context):
            query_tokenized.extend(self.query_tokenizer.tokenize(word))
            if self.add_spatial_tokens:
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
        assert "query_id" in baskets[0].raw and "start_index" in baskets[0].raw and "end_index" in baskets[0].raw, "Invalid sample"
        for basket in baskets:
            clear_text = {}
            tokenized = {}
            features = [{}]  # type: ignore
            # extract query, positive context passages and titles, hard-negative passages and titles
            if "query" in basket.raw:
                try:
                    query_obj = {
                        "id": basket.raw["query_id"],
                        "goldChain": None,
                        "mention": basket.raw["query_mention"],
                        "startIndex": basket.raw["start_index"],
                        "endIndex": basket.raw["end_index"],
                        "context": basket.raw["query"].split(" "),
                        "positive_examples": None,
                        "negative_examples": None,
                        "bm25_query": None,
                        "dummy": False,
                    }
                    query_feat, tokenized_query = self.get_query_feat(TrainExample(query_obj))
                    if len(tokenized_query) == 0:
                        logger.warning(
                            f"The query could not be tokenized, likely because it contains a character that the query tokenizer does not recognize")
                        return None

                    clear_text["query_text"] = " ".join(tokenized_query)
                    tokenized["query_tokens"] = tokenized_query
                    features[0]["query_input_ids"] = query_feat.input_ids
                    features[0]["query_segment_ids"] = query_feat.segment_ids
                    features[0]["query_attention_mask"] = query_feat.input_mask
                    features[0]["query_start"] = query_feat.query_event_start
                    features[0]["query_end"] = query_feat.query_event_end
                except Exception as e:
                    features = None  # type: ignore

            sample = Sample(id="",
                            clear_text=clear_text,
                            tokenized=tokenized,
                            features=features)  # type: ignore
            basket.samples = [sample]
        return baskets
