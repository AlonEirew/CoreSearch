from typing import List

from haystack import Document

from src.utils import io_utils, dpr_utils


def faiss_index(documents: List[Document],
                faiss_file_path,
                sql_rul,
                query_encode,
                passage_encode,
                infer_tokenizer_classes):

    document_store, retriever = dpr_utils.create_faiss_dpr(sql_rul,
                                                           query_encode,
                                                           passage_encode,
                                                           infer_tokenizer_classes)
    document_store.delete_documents()
    print("Writing document to FAISS index (may take a while)..")
    document_store.write_documents(documents=documents)
    document_store.update_embeddings(retriever=retriever)
    document_store.save(faiss_file_path)
    print("FAISS index creation done!")


def main():
    documents = io_utils.read_wec_to_haystack_doc_list("data/resources/WEC-ES/Dev_all_passages.json")
    # documents = io_utils.read_wec_to_haystack_doc_list("data/resources/train/Dev_training_passages.json")
    faiss_path_prefix = "weces_index_multiset/weces_dev_index"
    faiss_file_path = "%s.faiss" % faiss_path_prefix
    sql_rul = "sqlite:///%s.db" % faiss_path_prefix
    # query_encode = "bert-base-cased"
    # passage_encode = "bert-base-cased"
    # query_encode = "SpanBERT/spanbert-base-cased"
    # passage_encode = "SpanBERT/spanbert-base-cased"
    query_encode = "facebook/dpr-question_encoder-multiset-base"
    passage_encode = "facebook/dpr-ctx_encoder-multiset-base"
    infer_tokenizer_classes = False
    faiss_index(documents, faiss_file_path, sql_rul, query_encode, passage_encode, infer_tokenizer_classes)


if __name__ == '__main__':
    main()
