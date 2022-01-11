import logging
from pathlib import Path
from typing import Union, Optional, List

import torch
from farm.modeling.biadaptive_model import BiAdaptiveModel
from farm.modeling.prediction_head import TextSimilarityHead
from haystack.document_stores import BaseDocumentStore
from haystack.nodes import DensePassageRetriever
from torch.nn import DataParallel

from src.override_classes.wec_text_processor import WECSimilarityProcessor

logger = logging.getLogger(__name__)


class WECDensePassageRetriever(DensePassageRetriever):
    """
        Retriever that uses a bi-encoder (one transformer for query, one transformer for passage).
        See the original paper for more details:
        Karpukhin, Vladimir, et al. (2020): "Dense Passage Retrieval for Open-Domain Question Answering."
        (https://arxiv.org/abs/2004.04906).
    """

    def __init__(self,
                 document_store: BaseDocumentStore,
                 query_embedding_model: Union[Path, str] = "facebook/dpr-question_encoder-single-nq-base",
                 passage_embedding_model: Union[Path, str] = "facebook/dpr-ctx_encoder-single-nq-base",
                 model_version: Optional[str] = None,
                 max_seq_len_query: int = 64,
                 max_seq_len_passage: int = 256,
                 top_k: int = 10,
                 use_gpu: bool = True,
                 batch_size: int = 16,
                 embed_title: bool = True,
                 use_fast_tokenizers: bool = True,
                 infer_tokenizer_classes: bool = False,
                 similarity_function: str = "dot_product",
                 global_loss_buffer_size: int = 150000,
                 progress_bar: bool = True,
                 devices: Optional[List[Union[int, str, torch.device]]] = None
                 ):

        super(WECDensePassageRetriever, self).__init__(
            document_store,
            query_embedding_model,
            passage_embedding_model,
            model_version,
            max_seq_len_query,
            max_seq_len_passage,
            top_k,
            use_gpu,
            batch_size,
            embed_title,
            use_fast_tokenizers,
            infer_tokenizer_classes,
            similarity_function,
            global_loss_buffer_size,
            progress_bar,
            devices
        )

        self.processor = WECSimilarityProcessor(query_tokenizer=self.query_tokenizer,
                                                passage_tokenizer=self.passage_tokenizer,
                                                max_seq_len_passage=max_seq_len_passage,
                                                max_seq_len_query=max_seq_len_query,
                                                label_list=["hard_negative", "positive"],
                                                metric="text_similarity_metric",
                                                embed_title=embed_title,
                                                num_hard_negatives=0,
                                                num_positives=1)

        prediction_head = TextSimilarityHead(similarity_function=similarity_function, global_loss_buffer_size=global_loss_buffer_size)
        self.model = BiAdaptiveModel(
            language_model1=self.query_encoder,
            language_model2=self.passage_encoder,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=0.1,
            lm1_output_types=["per_sequence"],
            lm2_output_types=["per_sequence"],
            device=self.devices[0],
        )

        self.model.connect_heads_with_processor(self.processor.tasks, require_labels=False)

        if len(self.devices) > 1:
            self.model = DataParallel(self.model, device_ids=self.devices)
