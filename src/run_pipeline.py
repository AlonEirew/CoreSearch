from typing import List

from haystack.reader import FARMReader

from src.data_obj import Query, Cluster, TrainExample
from src.index import faiss_index, elastic_index
from src.pipeline.pipelines import QAPipeline, RetrievalOnlyPipeline
from src.utils import io_utils, measurments, data_utils


def main():
    # index_type = "elastic_bm25"
    index_type = "faiss_dpr"
    run_pipe_str = "retriever"
    # run_pipe_str = "qa"
    es_index = SPLIT.lower()

    faiss_index_file = "wec_" + es_index + "_index.faiss"
    sql_url = "sqlite:///weces_" + es_index + ".db"

    golds: List[Cluster] = io_utils.read_gold_file("resources/WEC-ES/" + SPLIT + "_gold_clusters.json")
    # query_examples: List[Query] = io_utils.read_query_file("resources/WEC-ES/" + SPLIT + "_queries.json")
    query_examples: List[TrainExample] = io_utils.read_train_example_file("resources/train/" + SPLIT + "_training_queries.json")
    for query in query_examples:
        query.context = query.bm25_query.split(" ")

    if index_type == "faiss_dpr":
        document_store, retriever = faiss_index.load_faiss_dpr(faiss_index_file, sql_url)
    elif index_type == "elastic_bm25":
        document_store, retriever = elastic_index.load_elastic_bm25(es_index)
    else:
        raise TypeError

    print(index_type + " Document store and retriever created..")
    print("Total indexed documents to be searched=" + str(document_store.get_document_count()))

    if run_pipe_str == "qa":
        pipeline = QAPipeline(document_store=document_store,
                              retriever=retriever,
                              reader=FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True),
                              ret_topk=10,
                              read_topk=5)
    elif run_pipe_str == "retriever":
        pipeline = RetrievalOnlyPipeline(document_store=document_store,
                                         retriever=retriever,
                                         ret_topk=150)
    else:
        raise TypeError

    print("Running " + run_pipe_str + " pipeline..")
    predict_and_eval(pipeline, golds, query_examples)


def predict_and_eval(pipeline, golds, query_examples):
    predictions = pipeline.run_end_to_end(query_examples=query_examples)
    predictions_arranged = data_utils.query_results_to_ids_list(predictions)
    golds_arranged = data_utils.clusters_to_ids_list(gold_clusters=golds)

    assert len(predictions) == len(golds_arranged)
    # print("HIT@10=" + str(measurments.hit_rate(predictions=predictions_arranged, golds=golds_arranged, topk=10)))
    print("MRR@10=" + str(measurments.mean_reciprocal_rank(predictions=predictions_arranged, golds=golds_arranged, topk=10)))
    print("RECALL@10=" + str(measurments.recall(predictions=predictions_arranged, golds=golds_arranged, topk=10)))
    print("RECALL@50=" + str(measurments.recall(predictions=predictions_arranged, golds=golds_arranged, topk=50)))
    print("RECALL@100=" + str(measurments.recall(predictions=predictions_arranged, golds=golds_arranged, topk=100)))


if __name__ == '__main__':
    SPLIT = "Dev"
    main()
