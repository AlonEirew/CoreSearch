from haystack.document_store import FAISSDocumentStore

from src.utils import io_utils


def faiss_index(documents):
    document_store = FAISSDocumentStore(sql_url="sqlite:///weces_train.db",
                                        faiss_index_factory_str="Flat",
                                        similarity="dot_product")
    document_store.delete_documents()
    document_store.write_documents(documents=documents)
    document_store.save("wec_train_index.faiss")
    print("FAISS index creation done!")


def main():
    documents = io_utils.read_wec_to_haystack_doc_list("resources/train/wec_es_Train_passages_segment.json")
    faiss_index(documents)


if __name__ == '__main__':
    main()
