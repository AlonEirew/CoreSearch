import logging
from typing import List, Dict


from src.data_obj import Passage, Cluster, TrainExample, QueryResult
from src.pipeline.pipelines import RetrievalOnlyPipeline

from src.index import elastic_index
from src.pipeline.run_e2e_pipeline import generate_query_text, print_results
from src.utils import io_utils, data_utils

SPLIT = "Dev"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
    es_index = SPLIT.lower()
    ret_top_k = 500

    gold_cluster_file = "data/resources/WEC-ES/clean/" + SPLIT + "_gold_clusters.json"
    queries_file = "data/resources/WEC-ES/train/" + SPLIT + "_queries.json"
    # passages_file = "data/resources/WEC-ES/clean/" + SPLIT + "_all_passages.json"

    result_out_file = "results/" + es_index + "_elastic_query_bm25.txt"

    golds: List[Cluster] = io_utils.read_gold_file(gold_cluster_file)
    query_examples: List[TrainExample] = io_utils.read_train_example_file(queries_file)

    # passage_examples: List[Passage] = io_utils.read_passages_file(passages_file)
    # passage_dict: Dict[str, Passage] = {obj.id: obj for obj in passage_examples}

    query_examples = generate_query_text(None, query_examples, "bm25", "all")

    document_store, retriever = elastic_index.load_elastic_bm25(es_index)
    logger.info("ElasticSearch Document store and retriever objects created..")
    logger.info("Total indexed documents to be searched=" + str(document_store.get_document_count()))
    pipeline = RetrievalOnlyPipeline(document_store=document_store,
                                     retriever=retriever,
                                     ret_topk=ret_top_k)

    logger.info("Running QA pipeline..")
    predict_and_eval(pipeline, golds, query_examples, "retriever", result_out_file)


def predict_and_eval(pipeline, golds, query_examples, run_pipe_str, result_out_file):
    predictions: List[QueryResult] = pipeline.run_end_to_end(query_examples=query_examples, query_as_dict=False)
    golds_arranged = data_utils.clusters_to_ids_list(gold_clusters=golds)

    # assert len(predictions) == len(golds_arranged)
    print_results(predictions, golds_arranged, run_pipe_str, result_out_file)


if __name__ == '__main__':
    main()
