import logging
from pathlib import Path
from typing import Union, Optional, List, Type

import torch
from haystack.document_stores import BaseDocumentStore
from haystack.modeling.model.biadaptive_model import BiAdaptiveModel
from haystack.modeling.model.prediction_head import TextSimilarityHead
from haystack.modeling.model.tokenization import Tokenizer
from haystack.modeling.utils import initialize_device_settings
from haystack.nodes import DensePassageRetriever
from torch.nn import DataParallel

from src.override_classes.override_language_model import OverrideLanguageModel
from src.override_classes.wec_processor import WECSimilarityProcessor

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


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
                 devices: Optional[List[Union[int, str, torch.device]]] = None,
                 use_auth_token: Optional[Union[str, bool]] = None,
                 processor_type: Type[WECSimilarityProcessor] = None,
                 add_spatial_tokens: bool = False
                 ):

        # save init parameters to enable export of component config as YAML
        self.set_config(
            document_store=document_store, query_embedding_model=query_embedding_model,
            passage_embedding_model=passage_embedding_model,
            model_version=model_version, max_seq_len_query=max_seq_len_query, max_seq_len_passage=max_seq_len_passage,
            top_k=top_k, use_gpu=use_gpu, batch_size=batch_size, embed_title=embed_title,
            use_fast_tokenizers=use_fast_tokenizers, infer_tokenizer_classes=infer_tokenizer_classes,
            similarity_function=similarity_function, progress_bar=progress_bar, devices=devices,
            use_auth_token=use_auth_token
        )

        if devices is not None:
            self.devices = devices
        else:
            self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)

        if batch_size < len(self.devices):
            logger.warning("Batch size is less than the number of devices. All gpus will not be utilized.")

        self.document_store = document_store
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.top_k = top_k

        if document_store is None:
            logger.warning("DensePassageRetriever initialized without a document store. "
                           "This is fine if you are performing DPR training. "
                           "Otherwise, please provide a document store in the constructor.")
        elif document_store.similarity != "dot_product":
            logger.warning(
                f"You are using a Dense Passage Retriever model with the {document_store.similarity} function. "
                "We recommend you use dot_product instead. "
                "This can be set when initializing the DocumentStore")

        self.infer_tokenizer_classes = infer_tokenizer_classes
        tokenizers_default_classes = {
            "query": "DPRQuestionEncoderTokenizer",
            "passage": "DPRContextEncoderTokenizer"
        }
        if self.infer_tokenizer_classes:
            tokenizers_default_classes["query"] = None  # type: ignore
            tokenizers_default_classes["passage"] = None  # type: ignore

        self.query_encoder = OverrideLanguageModel.load(pretrained_model_name_or_path=query_embedding_model,
                                                        language_model_class="WECQuestionEncoder")

        self.query_tokenizer = Tokenizer.load(pretrained_model_name_or_path=query_embedding_model,
                                              revision=model_version,
                                              do_lower_case=True,
                                              use_fast=use_fast_tokenizers,
                                              tokenizer_class=tokenizers_default_classes["query"],
                                              use_auth_token=use_auth_token)

        self.passage_encoder = OverrideLanguageModel.load(pretrained_model_name_or_path=passage_embedding_model,
                                                          language_model_class="WECContextEncoder")

        self.passage_tokenizer = Tokenizer.load(pretrained_model_name_or_path=passage_embedding_model,
                                                revision=model_version,
                                                do_lower_case=True,
                                                use_fast=use_fast_tokenizers,
                                                tokenizer_class=tokenizers_default_classes["passage"],
                                                use_auth_token=use_auth_token)

        self.processor = processor_type(query_tokenizer=self.query_tokenizer,
                                        passage_tokenizer=self.passage_tokenizer,
                                        max_seq_len_passage=max_seq_len_passage,
                                        max_seq_len_query=max_seq_len_query,
                                        label_list=["hard_negative", "positive"],
                                        metric="text_similarity_metric",
                                        embed_title=embed_title,
                                        num_hard_negatives=0,
                                        num_positives=1,
                                        add_spatial_tokens=add_spatial_tokens)

        if add_spatial_tokens:
            self.query_encoder.model.resize_token_embeddings(len(self.query_tokenizer))

        prediction_head = TextSimilarityHead(similarity_function=similarity_function,
                                             global_loss_buffer_size=global_loss_buffer_size)
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
