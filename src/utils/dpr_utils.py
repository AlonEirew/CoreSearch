from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever


def create_default_faiss_doc_store(sql_rul):
    return FAISSDocumentStore(sql_url=sql_rul,
                              similarity="dot_product")


def load_faiss_doc_store(faiss_file_path, faiss_config_file):
    return FAISSDocumentStore.load(index_path=faiss_file_path,
                                   config_path=faiss_config_file)


def create_default_dpr(document_store, query_encode, passage_encode, infer_tokenizer_classes=True):
    return DensePassageRetriever(document_store=document_store,
                                 query_embedding_model=query_encode,
                                 passage_embedding_model=passage_encode,
                                 infer_tokenizer_classes=infer_tokenizer_classes,
                                 max_seq_len_query=64,
                                 max_seq_len_passage=180,
                                 batch_size=16,
                                 use_gpu=True,
                                 embed_title=False,
                                 use_fast_tokenizers=False)


def load_dpr(retriever_model, document_store):
    return DensePassageRetriever.load(load_dir=retriever_model,
                                      document_store=document_store,
                                      infer_tokenizer_classes=True,
                                      use_fast_tokenizers=False)


def create_faiss_dpr(sql_rul, query_encode, passage_encode, infer_tokenizer_classes=True):
    document_store = create_default_faiss_doc_store(sql_rul)
    retriever = create_default_dpr(document_store, query_encode, passage_encode, infer_tokenizer_classes)
    return document_store, retriever


def load_faiss_dpr(faiss_file_path, faiss_config_file, retriever_model=None):
    document_store = load_faiss_doc_store(faiss_file_path, faiss_config_file)
    retriever = load_dpr(retriever_model, document_store)
    return document_store, retriever
