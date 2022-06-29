import logging

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever

from src.override_classes.file_doc_store import FileDocStore
from src.override_classes.retriever.search_dense import CoreSearchDensePassageRetriever

logger = logging.getLogger("dpr_utils")
logger.setLevel(logging.DEBUG)


def create_default_faiss_doc_store(sql_rul, similarity, faiss_index_factory_str):
    return FAISSDocumentStore(sql_url=sql_rul,
                              similarity=similarity,
                              faiss_index_factory_str=faiss_index_factory_str)


def create_file_doc_store(result_file, passages_file):
    return FileDocStore(result_file=result_file, passages_file=passages_file)


def load_faiss_doc_store(faiss_file_path, faiss_config_file):
    return FAISSDocumentStore.load(index_path=faiss_file_path,
                                   config_path=faiss_config_file)


def load_dpr(document_store, query_encode, passage_encode, infer_tokenizer_classes,
             max_seq_len_query=64, max_seq_len_passage=180, batch_size=16):
    retriever = DensePassageRetriever(document_store=document_store, query_embedding_model=query_encode,
                                      passage_embedding_model=passage_encode,
                                      infer_tokenizer_classes=infer_tokenizer_classes,
                                      max_seq_len_query=max_seq_len_query, max_seq_len_passage=max_seq_len_passage,
                                      batch_size=batch_size, use_gpu=True, embed_title=False, use_fast_tokenizers=False)
    return retriever


def create_faiss_dpr(sql_rul, query_encode, passage_encode, infer_tokenizer_classes,
                     max_seq_len_query, max_seq_len_passage, batch_size, similarity, faiss_index_factory_str):
    document_store = create_default_faiss_doc_store(sql_rul, similarity, faiss_index_factory_str)
    retriever = load_dpr(document_store, query_encode, passage_encode,
                         infer_tokenizer_classes,
                         max_seq_len_query,
                         max_seq_len_passage,
                         batch_size)
    return document_store, retriever


def create_wec_faiss_dpr(sql_rul, query_encode, passage_encode, infer_tokenizer_classes,
                         max_seq_len_query, max_seq_len_passage, batch_size, similarity, processor_type,
                         add_special_tokens, faiss_index_factory_str):
    document_store = create_default_faiss_doc_store(sql_rul, similarity, faiss_index_factory_str)
    retriever = CoreSearchDensePassageRetriever(document_store=document_store, query_embedding_model=query_encode,
                                                passage_embedding_model=passage_encode,
                                                infer_tokenizer_classes=infer_tokenizer_classes,
                                                max_seq_len_query=max_seq_len_query,
                                                max_seq_len_passage=max_seq_len_passage,
                                                batch_size=batch_size, use_gpu=True, embed_title=False,
                                                use_fast_tokenizers=False, processor_type=processor_type,
                                                add_special_tokens=add_special_tokens)
    return document_store, retriever


def load_faiss_dpr(faiss_file_path, faiss_config_file, query_encode, passage_encode,
                   infer_tokenizer_classes, max_seq_len_query, max_seq_len_passage, batch_size):
    document_store = load_faiss_doc_store(faiss_file_path, faiss_config_file)
    retriever = load_dpr(document_store,
                         query_encode,
                         passage_encode,
                         infer_tokenizer_classes,
                         max_seq_len_query,
                         max_seq_len_passage,
                         batch_size)
    return document_store, retriever


def load_wec_faiss_dpr(result_file, passage_file, query_encode, passage_encode,
                       infer_tokenizer_classes, max_seq_len_query, max_seq_len_passage, batch_size,
                       processor_type, add_special_tokens):
    document_store = create_file_doc_store(result_file, passage_file)
    retriever = CoreSearchDensePassageRetriever(document_store=document_store, query_embedding_model=query_encode,
                                                passage_embedding_model=passage_encode,
                                                infer_tokenizer_classes=infer_tokenizer_classes,
                                                max_seq_len_query=max_seq_len_query,
                                                max_seq_len_passage=max_seq_len_passage,
                                                batch_size=batch_size, use_gpu=True, embed_title=False,
                                                use_fast_tokenizers=False, processor_type=processor_type,
                                                add_special_tokens=add_special_tokens)
    return document_store, retriever
