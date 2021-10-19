from haystack.document_store import ElasticsearchDocumentStore

from src.utils import io_utils


def elastic_index(documents):
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
    document_store.write_documents(documents)
    print("Elastic index creation done!")


def main():
    documents = io_utils.read_wec_to_haystack_doc_list("resources/train/wec_es_Train_passages_segment.json")
    elastic_index(documents)


if __name__ == '__main__':
    main()
