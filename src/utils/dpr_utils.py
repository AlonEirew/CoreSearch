import logging

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from transformers import BertTokenizer

from src.override_classes.wec_bm25_processor import WECBM25Processor
from src.override_classes.wec_dense import WECDensePassageRetriever

logger = logging.getLogger("dpr_utils")
logger.setLevel(logging.DEBUG)


def create_default_faiss_doc_store(sql_rul, similarity):
    return FAISSDocumentStore(sql_url=sql_rul,
                              similarity=similarity)


def load_faiss_doc_store(faiss_file_path, faiss_config_file):
    return FAISSDocumentStore.load(index_path=faiss_file_path,
                                   config_path=faiss_config_file)


def load_dpr(document_store, query_encode, passage_encode, infer_tokenizer_classes=True,
             max_seq_len_query=64, max_seq_len_passage=180, batch_size=16):
    retriever = DensePassageRetriever(document_store=document_store, query_embedding_model=query_encode,
                                      passage_embedding_model=passage_encode,
                                      infer_tokenizer_classes=infer_tokenizer_classes,
                                      max_seq_len_query=max_seq_len_query, max_seq_len_passage=max_seq_len_passage,
                                      batch_size=batch_size, use_gpu=True, embed_title=False, use_fast_tokenizers=False)
    return retriever


def create_faiss_dpr(sql_rul, query_encode, passage_encode, infer_tokenizer_classes,
                     max_seq_len_query, max_seq_len_passage, batch_size, similarity):
    document_store = create_default_faiss_doc_store(sql_rul, similarity)
    retriever = load_dpr(document_store, query_encode, passage_encode,
                         infer_tokenizer_classes,
                         max_seq_len_query,
                         max_seq_len_passage,
                         batch_size)
    return document_store, retriever


def create_wec_faiss_dpr(sql_rul, query_encode, passage_encode, infer_tokenizer_classes,
                         max_seq_len_query, max_seq_len_passage, batch_size, similarity, processor_type,
                         add_spatial_tokens):
    document_store = create_default_faiss_doc_store(sql_rul, similarity)
    retriever = WECDensePassageRetriever(document_store=document_store, query_embedding_model=query_encode,
                                         passage_embedding_model=passage_encode,
                                         infer_tokenizer_classes=infer_tokenizer_classes,
                                         max_seq_len_query=max_seq_len_query, max_seq_len_passage=max_seq_len_passage,
                                         batch_size=batch_size, use_gpu=True, embed_title=False,
                                         use_fast_tokenizers=False, processor_type=processor_type,
                                         add_spatial_tokens=add_spatial_tokens)
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


def load_wec_faiss_dpr(faiss_file_path, faiss_config_file, query_encode, passage_encode,
                       infer_tokenizer_classes, max_seq_len_query, max_seq_len_passage, batch_size,
                       processor_type, add_spatial_tokens):
    document_store = load_faiss_doc_store(faiss_file_path, faiss_config_file)
    retriever = WECDensePassageRetriever(document_store=document_store, query_embedding_model=query_encode,
                                         passage_embedding_model=passage_encode,
                                         infer_tokenizer_classes=infer_tokenizer_classes,
                                         max_seq_len_query=max_seq_len_query, max_seq_len_passage=max_seq_len_passage,
                                         batch_size=batch_size, use_gpu=True, embed_title=False,
                                         use_fast_tokenizers=False, processor_type=processor_type,
                                         add_spatial_tokens=add_spatial_tokens)
    return document_store, retriever
