"""
This script will create a new ElasticSearch index containing documents generated from the input file.
In case given index already exists, it will be deleted by this process and recreated.
Usage:
    elastic_index.py --input=<PassageFile> --index=<IndexName>

Options:
    -h --help                   Show this screen.
    --input=<PassageFile>       input file containing all passages to be indexed into ElasticSearch
    --index=<IndexName>         The index name to create in ElasticSearch

"""
from typing import List

from docopt import docopt

from haystack import Document
from haystack.nodes import ElasticsearchRetriever
from src.index.wec_elasticsearch import WECElasticsearchDocumentStore
from src.override_classes.retriever.wec_sparse import WECElasticsearchRetriever
from src.utils import io_utils


def load_elastic_bm25(index):
    document_store = WECElasticsearchDocumentStore(index=index)
    retriever = ElasticsearchRetriever(document_store)
    return document_store, retriever


def load_wec_elastic_bm25(index):
    document_store = WECElasticsearchDocumentStore(index=index)
    retriever = WECElasticsearchRetriever(document_store)
    return document_store, retriever


def elastic_index(index: str, documents: List[Document]):
    document_store = WECElasticsearchDocumentStore(host="localhost", username="", password="", index=index)
    document_store.delete_documents()
    print("Writing document to Elastic...")
    document_store.write_documents(documents)
    print("Elastic index creation done!")
    print("Total indexed documents=" + str(document_store.get_document_count()))


def main(input_file, index):
    print("Reading input file and converting to haystack documents class..")
    documents = io_utils.read_wec_to_haystack_doc_list(input_file)
    elastic_index(index, documents)


if __name__ == '__main__':
    _arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    print(_arguments)
    _input_file = _arguments.get("--input")
    _index = _arguments.get("--index")
    main(_input_file, _index)
