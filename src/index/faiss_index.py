from typing import List

from haystack import Document
from haystack.document_store import FAISSDocumentStore
from haystack.retriever import DensePassageRetriever

from src.utils import io_utils


def load_faiss_dpr(faiss_file_path, sql_rul):
    document_store = FAISSDocumentStore.load(faiss_file_path=faiss_file_path,
                                             sql_url=sql_rul,
                                             index="document")

    retriever = get_dpr(document_store)
    return document_store, retriever


def get_dpr(document_store):
    return DensePassageRetriever(document_store=document_store,
                                 query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                 passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                 max_seq_len_query=64,
                                 max_seq_len_passage=256,
                                 batch_size=16,
                                 use_gpu=True,
                                 embed_title=False,
                                 use_fast_tokenizers=False)


def faiss_index(documents: List[Document], faiss_file_path, sql_rul):
    document_store = FAISSDocumentStore(sql_url=sql_rul,
                                        faiss_index_factory_str="Flat",
                                        similarity="dot_product")

    retriever = get_dpr(document_store)

    document_store.delete_documents()
    print("Writing document to FAISS index (may take a while)..")
    document_store.write_documents(documents=documents)
    document_store.update_embeddings(retriever=retriever)
    document_store.save(faiss_file_path)
    print("FAISS index creation done!")


def main():
    documents = io_utils.read_wec_to_haystack_doc_list("resources/WEC-ES/Dev_passages.json")
    faiss_file_path = "wec_dev_index.faiss"
    sql_rul = "sqlite:///weces_dev.db"
    faiss_index(documents, faiss_file_path, sql_rul)


if __name__ == '__main__':
    main()
