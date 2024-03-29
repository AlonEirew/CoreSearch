import json
import logging
from pathlib import Path
from typing import Union, Optional, List, Type, Dict

import torch
from haystack import Document
from haystack.document_stores import BaseDocumentStore
from haystack.modeling.model.biadaptive_model import BiAdaptiveModel
from haystack.modeling.model.prediction_head import TextSimilarityHead
from haystack.modeling.model.tokenization import Tokenizer
from haystack.modeling.utils import initialize_device_settings
from haystack.nodes import DensePassageRetriever
from torch.nn import DataParallel

from src.override_classes.file_doc_store import FileDocStore
from src.override_classes.override_language_model import OverrideLanguageModel
from src.override_classes.retriever.search_processor import CoreSearchSimilarityProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CoreSearchDensePassageRetriever(DensePassageRetriever):
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
                 processor_type: Type[CoreSearchSimilarityProcessor] = None,
                 add_special_tokens: bool = False
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
                                                        language_model_class="CoreSearchQuestionEncoder")

        self.query_tokenizer = Tokenizer.load(pretrained_model_name_or_path=query_embedding_model,
                                              revision=model_version,
                                              do_lower_case=True,
                                              use_fast=use_fast_tokenizers,
                                              tokenizer_class=tokenizers_default_classes["query"],
                                              use_auth_token=use_auth_token)

        self.passage_encoder = OverrideLanguageModel.load(pretrained_model_name_or_path=passage_embedding_model,
                                                          language_model_class="CoreSearchContextEncoder")

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
                                        add_special_tokens=add_special_tokens)

        if add_special_tokens:
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

    def retrieve(self, query: Dict, filters: dict = None, top_k: Optional[int] = None, index: str = None,
                 headers: Optional[Dict[str, str]] = None) -> List[Document]:
        if top_k is None:
            top_k = self.top_k
        if not self.document_store:
            logger.error("Cannot perform retrieve() since DensePassageRetriever initialized with document_store=None")
            return []

        if isinstance(query, str):
            query = json.loads(query)

        logger.info(f"Retrieving results for query-{json.dumps(query)}")
        # adding 1 to top-k in case one of the returned results is the same as query and need to be removed
        if isinstance(self.document_store, FileDocStore):
            documents = self.document_store.get_passages(query_id=query["query_id"], top_k=top_k+1)
        else:
            query_emb = self.embed_queries(texts=[query])
            documents = self.document_store.query_by_embedding(query_emb=query_emb[0], top_k=top_k+1, filters=filters,
                                                               index=index, headers=headers)

        documents = [doc for doc in documents if not doc.id == query['query_id']]
        return documents[:top_k]

    @staticmethod
    def predict_pairwise_cosine(query_rep, passage_rep):
        prediction = torch.cosine_similarity(query_rep, passage_rep)
        # prediction = torch.round(prediction)
        return prediction

    @staticmethod
    def predict_pairwise_dot_product(query_rep, passage_rep):
        # prediction = torch.dot(query_rep, passage_rep)
        prediction = query_rep @ passage_rep.T
        # prediction = torch.round(prediction)
        return prediction.squeeze()
