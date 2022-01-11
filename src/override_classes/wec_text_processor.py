import logging
import logging
import random
from typing import Tuple, List

import numpy as np
from farm.data_handler.dataset import convert_features_to_dataset
from farm.data_handler.processor import TextSimilarityProcessor
from farm.data_handler.samples import Sample
from tqdm import tqdm

from src.data_obj import BasicMent, TrainExample
from src.utils import io_utils

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
        label_list=None
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

    def file_to_dicts(self, files: str) -> [dict]:
        """
        Converts a Dense Passage Retrieval (DPR) data file in json format to a list of dictionaries.

        :param file: filename of DPR data in json format
                Each sample is a dictionary of format:
                {"dataset": str,
                "question": str,
                "answers": list of str
                "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
                "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
                "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
                }


        Returns:
        list of dictionaries: List[dict]
            each dictionary:
            {"query": str,
            "passages": [{"text": document_text, "title": xxx, "label": "positive", "external_id": abb123},
            {"text": document_text, "title": xxx, "label": "hard_negative", "external_id": abb134},
            ...]}
        """

        queries_file, passages_file = str(files).split("#")
        dicts = self.read_wec(queries_file, passages_file)

        # shuffle dicts to make sure that similar positive passages do not end up in one batch
        dicts = random.sample(dicts, len(dicts))
        return dicts

    def read_wec(self, queries_file, passages_file):
        query_examples: List[TrainExample] = io_utils.read_train_example_file(queries_file)
        passages_examples = io_utils.read_passages_file(passages_file)
        passages_examples = {passage.id: passage for passage in passages_examples}
        logger.info("Done loading examples file, queries-" + queries_file + ", passages-" + passages_file)
        logger.info("Total examples loaded, queries=" + str(len(query_examples)) + ", passages=" + str(len(passages_examples)))
        logger.info("Starting to generate examples...")
        standard_dicts = []
        for query_obj in tqdm(query_examples, "Loading Queries"):
            sample = {}
            passages = []
            sample["query"] = " ".join(query_obj.context)
            sample["query_obj"] = self.convert_query_to_dpr_obj(query_obj)
            for pos_id in query_obj.positive_examples[:self.num_positives]:
                passages.append(self.convert_to_dpr_obj(passages_examples[pos_id], "positive"))

            for neg_id in query_obj.negative_examples[:self.num_hard_negatives]:
                passages.append(self.convert_to_dpr_obj(passages_examples[neg_id], "hard_negative"))

            sample["passages"] = passages
            standard_dicts.append(sample)

        return standard_dicts

    @staticmethod
    def convert_to_dpr_obj(basic_mention: BasicMent, polarity):
        return {
            "title": "NA",
            "text": " ".join(basic_mention.context),
            "label": polarity,
            "external_id": basic_mention.id,
            "start_idx": basic_mention.startIndex,
            "end_idx": basic_mention.endIndex,
            "gold_chain": basic_mention.goldChain,
            "mention": basic_mention.mention
        }

    @staticmethod
    def convert_query_to_dpr_obj(basic_mention: BasicMent):
        return {
            "external_id": basic_mention.id,
            "start_idx": basic_mention.startIndex,
            "end_idx": basic_mention.endIndex,
            "gold_chain": basic_mention.goldChain,
            "mention": basic_mention.mention
        }

    def _convert_queries(self, baskets):
        for basket in baskets:
            clear_text = {}
            tokenized = {}
            features = [{}]
            # extract query, positive context passages and titles, hard-negative passages and titles
            if "query" in basket.raw:
                try:
                    query = self._normalize_question(basket.raw["query"])

                    # featurize the query
                    query_inputs = self.query_tokenizer.encode_plus(
                        text=query,
                        max_length=self.max_seq_len_query,
                        add_special_tokens=True,
                        truncation=True,
                        truncation_strategy='longest_first',
                        padding="max_length",
                        return_token_type_ids=True,
                    )

                    # tokenize query
                    tokenized_query = self.query_tokenizer.convert_ids_to_tokens(query_inputs["input_ids"])

                    if len(tokenized_query) == 0:
                        logger.warning(
                            f"The query could not be tokenized, likely because it contains a character that the query tokenizer does not recognize")
                        return None

                    clear_text["query_text"] = query
                    tokenized["query_tokens"] = tokenized_query
                    features[0]["query_input_ids"] = query_inputs["input_ids"]
                    features[0]["query_segment_ids"] = query_inputs["token_type_ids"]
                    features[0]["query_attention_mask"] = query_inputs["attention_mask"]
                except Exception as e:
                    features = None

            sample = Sample(id=None,
                            clear_text=clear_text,
                            tokenized=tokenized,
                            features=features)
            basket.samples = [sample]
        return baskets

    def _convert_contexts(self, baskets):
        for basket in baskets:
            if "passages" in basket.raw:
                try:
                    positive_context = list(filter(lambda x: x["label"] == "positive", basket.raw["passages"]))
                    if self.shuffle_positives:
                        random.shuffle(positive_context)
                    positive_context = positive_context[:self.num_positives]
                    hard_negative_context = list(filter(lambda x: x["label"] == "hard_negative", basket.raw["passages"]))
                    if self.shuffle_negatives:
                        random.shuffle(hard_negative_context)
                    hard_negative_context = hard_negative_context[:self.num_hard_negatives]

                    positive_ctx_titles = [passage.get("title", None) for passage in positive_context]
                    positive_ctx_texts = [passage["text"] for passage in positive_context]
                    hard_negative_ctx_titles = [passage.get("title", None) for passage in hard_negative_context]
                    hard_negative_ctx_texts = [passage["text"] for passage in hard_negative_context]

                    # all context passages and labels: 1 for positive context and 0 for hard-negative context
                    ctx_label = [1] * self.num_positives + [0] * self.num_hard_negatives
                    # featurize context passages
                    if self.embed_title:
                        # concatenate title with positive context passages + negative context passages
                        all_ctx = self._combine_title_context(positive_ctx_titles, positive_ctx_texts) + \
                                  self._combine_title_context(hard_negative_ctx_titles, hard_negative_ctx_texts)
                    else:
                        all_ctx = positive_ctx_texts + hard_negative_ctx_texts

                    # assign empty string tuples if hard_negative passages less than num_hard_negatives
                    all_ctx += [('', '')] * ((self.num_positives + self.num_hard_negatives) - len(all_ctx))

                    ctx_inputs = self.passage_tokenizer.batch_encode_plus(
                        all_ctx,
                        add_special_tokens=True,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_seq_len_passage,
                        return_token_type_ids=True
                    )

                    # TODO check if we need this and potentially remove
                    ctx_segment_ids = np.zeros_like(ctx_inputs["token_type_ids"], dtype=np.int32)

                    # get tokens in string format
                    tokenized_passage = [self.passage_tokenizer.convert_ids_to_tokens(ctx) for ctx in ctx_inputs["input_ids"]]

                    # for DPR we only have one sample containing query and corresponding (multiple) context features
                    sample = basket.samples[0]
                    sample.clear_text["passages"] = positive_context + hard_negative_context
                    sample.tokenized["passages_tokens"] = tokenized_passage
                    sample.features[0]["passage_input_ids"] = ctx_inputs["input_ids"]
                    sample.features[0]["passage_segment_ids"] = ctx_segment_ids
                    sample.features[0]["passage_attention_mask"] = ctx_inputs["attention_mask"]
                    sample.features[0]["label_ids"] = ctx_label
                except Exception as e:
                    basket.samples[0].features = None

        return baskets

    def _create_dataset(self, baskets):
        """
        Convert python features into pytorch dataset.
        Also removes potential errors during preprocessing.
        Flattens nested basket structure to create a flat list of features
        """
        features_flat = []
        basket_to_remove = []
        problematic_ids = set()
        for basket in baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:
                    features_flat.extend(sample.features)
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        if len(basket_to_remove) > 0:
            for basket in basket_to_remove:
                # if basket_to_remove is not empty remove the related baskets
                problematic_ids.add(basket.id_internal)
                baskets.remove(basket)

        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names, problematic_ids, baskets
