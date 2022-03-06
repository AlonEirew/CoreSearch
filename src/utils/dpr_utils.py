import logging

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from transformers import BertTokenizer

logger = logging.getLogger("dpr_utils")
logger.setLevel(logging.DEBUG)


def create_default_faiss_doc_store(sql_rul):
    return FAISSDocumentStore(sql_url=sql_rul,
                              similarity="dot_product")


def load_faiss_doc_store(faiss_file_path, faiss_config_file):
    return FAISSDocumentStore.load(index_path=faiss_file_path,
                                   config_path=faiss_config_file)


def load_dpr(document_store, query_encode, passage_encode, infer_tokenizer_classes=True,
             max_seq_len_query=64, max_seq_len_passage=180, batch_size=16, load_tokenizer=False):
    retriever = DensePassageRetriever(document_store=document_store, query_embedding_model=query_encode,
                                      passage_embedding_model=passage_encode,
                                      infer_tokenizer_classes=infer_tokenizer_classes,
                                      max_seq_len_query=max_seq_len_query, max_seq_len_passage=max_seq_len_passage,
                                      batch_size=batch_size, use_gpu=True, embed_title=False, use_fast_tokenizers=False)

    if load_tokenizer:
        logger.info("Replacing with model tokenizers from-" + query_encode + ", and " + passage_encode)
        retriever.query_tokenizer = BertTokenizer.from_pretrained(query_encode)
        retriever.passage_tokenizer = BertTokenizer.from_pretrained(passage_encode)

    return retriever


def create_faiss_dpr(sql_rul, query_encode, passage_encode, infer_tokenizer_classes,
                     max_seq_len_query, max_seq_len_passage, batch_size, load_tokenizer):
    document_store = create_default_faiss_doc_store(sql_rul)
    retriever = load_dpr(document_store, query_encode, passage_encode,
                         infer_tokenizer_classes,
                         max_seq_len_query,
                         max_seq_len_passage,
                         batch_size,
                         load_tokenizer)
    return document_store, retriever


def load_faiss_dpr(faiss_file_path, faiss_config_file, query_encode, passage_encode,
                   infer_tokenizer_classes, max_seq_len_query, max_seq_len_passage, batch_size, load_tokenizer):
    document_store = load_faiss_doc_store(faiss_file_path, faiss_config_file)
    retriever = load_dpr(document_store,
                         query_encode,
                         passage_encode,
                         infer_tokenizer_classes,
                         max_seq_len_query,
                         max_seq_len_passage,
                         batch_size,
                         load_tokenizer)
    return document_store, retriever
