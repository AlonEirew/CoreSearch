import logging
from typing import List

from haystack.modeling.data_handler.processor import TextSimilarityProcessor
from haystack.modeling.data_handler.samples import SampleBasket, Sample

from src.data_obj import TrainExample

logger = logging.getLogger(__name__)


class WECSimilarityProcessor(TextSimilarityProcessor):
    """
    Used to handle the Dense Passage Retrieval (DPR) datasets that come in json format, example: biencoder-nq-train.json, biencoder-nq-dev.json, trivia-train.json, trivia-dev.json

    Datasets can be downloaded from the official DPR github repository (https://github.com/facebookresearch/DPR)
    dataset format: list of dictionaries with keys: 'dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'
    Each sample is a dictionary of format:
    {"dataset": str,
    "question": str,
    "answers": list of str
    "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    }

    """
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
        tokenization=None
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
        self.tokenization = tokenization

    def _convert_queries(self, baskets: List[SampleBasket]):
        for basket in baskets:
            clear_text = {}
            tokenized = {}
            features = [{}]  # type: ignore
            # extract query, positive context passages and titles, hard-negative passages and titles
            if "query" in basket.raw:
                try:
                    query_obj = {"context": basket.raw["query"].split(" "), "dummy": True}
                    query_feat = self.tokenization.get_query_feat(TrainExample(query_obj), self.max_seq_len_query, False)

                    # tokenize query
                    tokenized_query = self.query_tokenizer.convert_ids_to_tokens(query_feat.query_input_ids)

                    if len(tokenized_query) == 0:
                        logger.warning(
                            f"The query could not be tokenized, likely because it contains a character that the query tokenizer does not recognize")
                        return None

                    clear_text["query_text"] = " ".join(tokenized_query)
                    tokenized["query_tokens"] = tokenized_query
                    features[0]["query_input_ids"] = query_feat.query_input_ids
                    features[0]["query_segment_ids"] = query_feat.query_segment_ids
                    features[0]["query_attention_mask"] = query_feat.query_input_mask
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
