"""
This script will create a new ElasticSearch index containing documents generated from the input file.
In case given index already exists, it will be deleted by this process and recreated.
Usage:
    elastic_index.py --input=<PassageFolder> --index=<IndexName>

Options:
    -h --help                   Show this screen.
    --input=<PassageFolder>     input folder containing all files to be indexed into ElasticSearch
    --index=<IndexName>         The index name to create in ElasticSearch

"""

import json
from os import listdir
from os.path import isfile, join
from typing import List, Dict, Any, Tuple

from docopt import docopt
from tqdm import tqdm

from haystack import Document
from src.index.search_elasticsearch import CoreSearchElasticsearchDocumentStore


def elastic_index(document_store, documents):
    print("Writing document to Elastic...")
    document_store.write_documents(documents)
    print("Elastic index creation done!")
    print("Total indexed documents so far=" + str(document_store.get_document_count()))


def main(input_folder, index):
    print("Reading input file and converting to haystack documents class..")
    onlyfiles = [join(input_folder, f) for f in listdir(input_folder) if isfile(join(input_folder, f))]
    document_store = CoreSearchElasticsearchDocumentStore(host="localhost", username="", password="", index=index)
    document_store.delete_documents()
    passage_id = 1
    for file in onlyfiles:
        print(f"Reading and indexing file-{file}")
        documents, passage_id = read_coresearch_to_haystack_doc_list(file, passage_id)
        elastic_index(document_store, documents)


def read_coresearch_to_haystack_doc_list(passages_file, passage_id) -> Tuple[List[Document], int]:
    with open(passages_file, 'r') as fis:
        raw_doc = json.load(fis)
    documents: List[Document] = []
    for doc in tqdm(raw_doc, desc="Converting passages"):
        in_doc_pass_id = 0
        for passage in doc['parsedParagraphs']:
            if len(passage.split(" ")) >= 40:
                meta: Dict[str, Any] = {'title': f"{doc['title']} ({in_doc_pass_id})"}
                documents.append(
                    Document(
                        content=passage,
                        id=passage_id,
                        meta=meta
                    )
                )
                passage_id += 1
                in_doc_pass_id += 1

    return documents, passage_id


if __name__ == '__main__':
    _arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    print(_arguments)
    _input_file = _arguments.get("--input")
    _index = _arguments.get("--index")
    main(_input_file, _index)
    print("Done!")
