from typing import Any, Dict

import os
import logging
from pathlib import Path

from haystack.pipelines.base import Pipeline
from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore

from src.override_classes.reader.wec_reader import WECReader
from src.override_classes.retriever.wec_context_processor import WECContextProcessor
from src.override_classes.retriever.wec_dense import WECDensePassageRetriever
from src.utils.dpr_utils import create_file_doc_store

from rest_api.controller.utils import RequestLimiter
from src.pipeline.pipelines import QAPipeline

logger = logging.getLogger(__name__)

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
    query_encode = checkpoint_dir + "dev_spanbert_hidden_cls_spatial_ctx_2it/query_encoder"
    passage_encode = checkpoint_dir + "dev_spanbert_hidden_cls_spatial_ctx_2it/passage_encoder"
    document_store = create_file_doc_store("file_indexes/dev_spanbert_hidden_cls_spatial_ctx_2it_top500.json",
                                           "data/resources/WEC-ES/Dev_all_passages.json")
    retriever = WECDensePassageRetriever(document_store=document_store, query_embedding_model=query_encode,
                                         passage_embedding_model=passage_encode,
                                         infer_tokenizer_classes=True,
                                         max_seq_len_query=64,
                                         max_seq_len_passage=180,
                                         batch_size=16, use_gpu=True, embed_title=False,
                                         use_fast_tokenizers=False, processor_type=WECContextProcessor,
                                         add_special_tokens=True)

    reader = WECReader(model_name_or_path="data/checkpoints/squad_roberta_ctx",
                       use_gpu=True, num_processes=8,
                       add_special_tokens=True)

    query_pipeline = Pipeline()
    # query_pipeline.add_node(component=document_store, name="DocumentStore", inputs=[])
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

    # logging.info(f"Loaded pipeline nodes: {query_pipeline.graph.nodes.keys()}") - Need Remove
    # END OF MY-CODE
    pipelines["query_pipeline"] = query_pipeline

    # Find document store
    document_store = query_pipeline.get_document_store()
    logging.info(f"Loaded docstore: {document_store}")
    pipelines["document_store"] = document_store

    # Setup concurrency limiter
    concurrency_limiter = RequestLimiter(config.CONCURRENT_REQUEST_PER_WORKER)
    logging.info("Concurrent requests per worker: {CONCURRENT_REQUEST_PER_WORKER}")
    pipelines["concurrency_limiter"] = concurrency_limiter

    # Load indexing pipeline (if available)
    try:
        indexing_pipeline = Pipeline.load_from_yaml(
            Path(config.PIPELINE_YAML_PATH), pipeline_name=config.INDEXING_PIPELINE_NAME
        )
        docstore = indexing_pipeline.get_document_store()
        if isinstance(docstore, UNSUPPORTED_DOC_STORES):
            indexing_pipeline = None
            raise EnvironmentError(
                "Indexing pipelines with FAISSDocumentStore or InMemoryDocumentStore are not supported by the REST APIs."
            )

    except EnvironmentError as e:
        indexing_pipeline = None
        logger.error(f"{e.message}\nFile Upload API will not be available.")

    finally:
        pipelines["indexing_pipeline"] = indexing_pipeline

    # Create directory for uploaded files
    os.makedirs(config.FILE_UPLOAD_PATH, exist_ok=True)

    return pipelines
