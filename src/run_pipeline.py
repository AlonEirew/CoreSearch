from typing import Dict

from haystack.document_store import FAISSDocumentStore, ElasticsearchDocumentStore
from haystack.reader import FARMReader
from haystack.retriever import DensePassageRetriever, ElasticsearchRetriever

from src.pipeline.pipelines import QAPipeline, RetrievalOnlyPipeline
from src.utils import io_utils


def get_faiss_dpr():
    document_store = FAISSDocumentStore.load(faiss_file_path="wec_train_index.faiss",
                                   sql_url="sqlite:///weces_train.db",
                                   index="document")

    retriever = DensePassageRetriever(document_store=document_store,
                                      query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                      passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                      max_seq_len_query=64,
                                      max_seq_len_passage=256,
                                      batch_size=16,
                                      use_gpu=True,
                                      embed_title=False,
                                      use_fast_tokenizers=False)

    document_store.update_embeddings(retriever=retriever)
    return document_store, retriever


def get_elastic_bm25():
    document_store = ElasticsearchDocumentStore()
    retriever = ElasticsearchRetriever(document_store)
    return document_store, retriever


def main():
    method_str = "faiss_dpr" #"elastic_bm25"
    run_pipe_str = "retriever"
    query_examples: Dict[int, Dict] = io_utils.read_query_examples_file("resources/train/wec_es_train_qsent_small.json")

    if method_str == "faiss_dpr":
        document_store, retriever = get_faiss_dpr()
    elif method_str == "elastic_bm25":
        document_store, retriever = get_elastic_bm25()
    else:
        raise TypeError

    if run_pipe_str == "qa":
        pipeline = QAPipeline(document_store=document_store,
                              retriever=retriever,
                              reader=FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True),
                              ret_topk=10,
                              read_topk=5)
    elif run_pipe_str == "retriever":
        pipeline = RetrievalOnlyPipeline(document_store=document_store,
                                         retriever=retriever,
                                         ret_topk=10)
        pipeline.run_end_to_end(query_examples=query_examples)
    else:
        raise TypeError

    pipeline.run_end_to_end(query_examples=query_examples)
    print("Total indexed documents searched=" + str(document_store.get_document_count()))


if __name__ == '__main__':
    main()
