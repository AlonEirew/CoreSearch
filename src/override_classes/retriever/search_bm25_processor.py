import logging
from typing import List, Tuple

from haystack.modeling.data_handler.samples import SampleBasket, Sample

from src.data_obj import TrainExample, QueryFeat
from src.override_classes.retriever.search_processor import CoreSearchSimilarityProcessor

logger = logging.getLogger(__name__)


class CoreSearchBM25Processor(CoreSearchSimilarityProcessor):
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
        super(CoreSearchBM25Processor, self).__init__(
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

        if self.add_special_tokens:
            raise ValueError("add_spatial_tokens flag is true in a BM25 tokenizer!")

    def _convert_queries(self, baskets: List[SampleBasket]):
        for basket in baskets:
            clear_text = {}
            tokenized = {}
            features = [{}]  # type: ignore
            # extract query, positive context passages and titles, hard-negative passages and titles
            if "query" in basket.raw:
                try:
                    query_obj = {"context": basket.raw["query"].split(" "), "dummy": True}
                    clear_text, tokenized, features, query_feat = self.tokenize_and_add_features(query_obj)
                except Exception as e:
                    features = None  # type: ignore

            sample = Sample(id="",
                            clear_text=clear_text,
                            tokenized=tokenized,
                            features=features)  # type: ignore
            basket.samples = [sample]
        return baskets

    def get_query_feat(self, query_obj: TrainExample) -> Tuple[QueryFeat, List[str]]:
        query_feat, tokenized_query = super().get_query_feat(query_obj)
        return query_feat, tokenized_query

    def tokenize_query(self, query_obj: TrainExample, max_query_length_exclude: int):
        query_tokenized = list()
        query_event_start_ind = query_event_end_ind = 0
        for index, word in enumerate(query_obj.context):
            query_tokenized.extend(self.query_tokenizer.tokenize(word))

        if len(query_tokenized) > max_query_length_exclude:
            query_tokenized = query_tokenized[0:max_query_length_exclude]

        query_input_mask = [1] * len(query_tokenized)
        while len(query_tokenized) < max_query_length_exclude:
            query_tokenized.append('[PAD]')
            query_input_mask.append(0)

        return query_event_start_ind, query_event_end_ind, query_tokenized, query_input_mask
