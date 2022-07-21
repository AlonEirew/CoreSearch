from typing import Any, Dict

import os
import logging

from src.override_classes.retriever.search_sparse import CoreSearchElasticsearchRetriever

from src.index.search_elasticsearch import CoreSearchElasticsearchDocumentStore

from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from src.override_classes.reader.search_reader import CoreSearchReader
from src.override_classes.retriever.search_context_processor import CoreSearchContextProcessor
from src.override_classes.retriever.search_dense import CoreSearchDensePassageRetriever
from src.pipeline.pipelines import QAPipeline

from src.utils.dpr_utils import create_file_doc_store, load_faiss_doc_store

from rest_api.controller.utils import RequestLimiter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Since each instance of FAISSDocumentStore creates an in-memory FAISS index, the Indexing & Query Pipelines would
# end up with different indices. The same applies for InMemoryDocumentStore.
UNSUPPORTED_DOC_STORES = (FAISSDocumentStore, InMemoryDocumentStore)


def setup_pipelines() -> Dict[str, Any]:
    # Re-import the configuration variables
    from rest_api import config  # pylint: disable=reimported

    pipelines = {}

    # Load query pipeline
    # query_pipeline = Pipeline.load_from_yaml(Path(config.PIPELINE_YAML_PATH), pipeline_name=config.QUERY_PIPELINE_NAME)

    # START OF MY-CODE
    checkpoint_dir = "data/checkpoints/"
    query_encode = checkpoint_dir + "Retriever_SpanBERT/0/query_encoder"
    passage_encode = checkpoint_dir + "Retriever_SpanBERT/0/passage_encoder"

    faiss_dir = "faiss_indexes" + "/best_index"
    faiss_path_prefix = faiss_dir + "/dev_test_index"
    faiss_file_path = "%s.faiss" % faiss_path_prefix
    faiss_config = "%s.json" % faiss_path_prefix

    elastic_index = "wiki"

    # document_store = create_file_doc_store("file_indexes/Retriever_SpanBERT_Best_it0_top500.json",
    #                                        "data/resources/CoreSearch/clean/Dev_all_passages.json")
    elastic_document_store = CoreSearchElasticsearchDocumentStore(index=elastic_index)
    sparse_retriever = CoreSearchElasticsearchRetriever(elastic_document_store)

    faiss_document_store = load_faiss_doc_store(faiss_file_path, faiss_config)
    dense_retriever = CoreSearchDensePassageRetriever(document_store=faiss_document_store,
                                                      query_embedding_model=query_encode,
                                                      passage_embedding_model=passage_encode,
                                                      infer_tokenizer_classes=True,
                                                      max_seq_len_query=64,
                                                      max_seq_len_passage=180,
                                                      batch_size=24, use_gpu=True, embed_title=False,
                                                      use_fast_tokenizers=False,
                                                      processor_type=CoreSearchContextProcessor,
                                                      add_special_tokens=True)

    reader = CoreSearchReader(model_name_or_path="data/checkpoints/Reader-RoBERTa_Kenton_Best/1",
                              use_gpu=True, num_processes=8, batch_size=24,
                              replace_prediction_heads=True, add_special_tokens=True,
                              prediction_head_str="kenton")

    dense_pipeline = QAPipeline(document_store=faiss_document_store,
                                retriever=dense_retriever,
                                reader=reader,
                                ret_topk=None,
                                read_topk=None)

    sparse_pipeline = QAPipeline(document_store=elastic_document_store,
                                 retriever=sparse_retriever,
                                 reader=reader,
                                 ret_topk=None,
                                 read_topk=None)

    pipelines["dense_pipeline"] = dense_pipeline
    pipelines["sparse_pipeline"] = sparse_pipeline

    # Setup concurrency limiter
    concurrency_limiter = RequestLimiter(config.CONCURRENT_REQUEST_PER_WORKER)
    logging.info("Concurrent requests per worker: {CONCURRENT_REQUEST_PER_WORKER}")
    pipelines["concurrency_limiter"] = concurrency_limiter

    pipelines["indexing_pipeline"] = None #indexing_pipeline

    # Create directory for uploaded files
    os.makedirs(config.FILE_UPLOAD_PATH, exist_ok=True)

    return pipelines
