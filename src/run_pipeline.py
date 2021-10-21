from typing import List

from haystack.reader import FARMReader

from src.data_obj import Query, Cluster
from src.index import faiss_index, elastic_index
from src.pipeline.pipelines import QAPipeline, RetrievalOnlyPipeline
from src.utils import io_utils, measurments, data_utils


def main():
    method_str = "elastic_bm25"
    run_pipe_str = "retriever"
    # method_str = "faiss_dpr"
    # run_pipe_str = "qa"
    query_examples: List[Query] = io_utils.read_query_file("resources/WEC-ES/Tiny_queries.json")
    golds: List[Cluster] = io_utils.read_gold_file("resources/WEC-ES/Train_gold_clusters.json")

    if method_str == "faiss_dpr":
        document_store, retriever = faiss_index.load_faiss_dpr()
    elif method_str == "elastic_bm25":
        document_store, retriever = elastic_index.load_elastic_bm25()
    else:
        raise TypeError

    print(method_str + " Document store and retriever created..")
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
                                         ret_topk=10)
    else:
        raise TypeError

    print("Running " + run_pipe_str + " pipeline..")
    predictions = pipeline.run_end_to_end(query_examples=query_examples)
    predictions_arranged = data_utils.query_results_to_ids_list(predictions)
    golds_arranged = data_utils.clusters_to_ids_list(gold_clusters=golds)

    print("MRR@10=" + str(measurments.mean_reciprocal_rank(predictions=predictions_arranged, golds=golds_arranged, topk=10)))
    print("HIT@10=" + str(measurments.hit_rate(predictions=predictions_arranged, golds=golds_arranged, topk=10)))


if __name__ == '__main__':
    main()
