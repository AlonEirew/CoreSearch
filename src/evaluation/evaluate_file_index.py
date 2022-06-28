from typing import List, Dict

from src.data_obj import TrainExample, Passage, QueryResult, Query, Cluster
from src.pipeline.run_e2e_pipeline import print_measurements
from src.utils import io_utils, data_utils
from src.utils.dpr_utils import create_file_doc_store


def main():
    index_file = "file_indexes/dev_spanbert_pooler_full_bm25_2it_top500.json"
    passages_file = "data/resources/WEC-ES/Dev_all_passages.json"
    queries_file = "data/resources/train/Dev_training_queries.json"
    gold_cluster_file = "data/resources/WEC-ES/" + SPLIT + "_gold_clusters.json"
    top_k = 500

    query_examples: List[TrainExample] = io_utils.read_train_example_file(queries_file)
    query_examples: Dict[str, TrainExample] = {query.id: query for query in query_examples}
    golds: List[Cluster] = io_utils.read_gold_file(gold_cluster_file)
    golds_arranged = data_utils.clusters_to_ids_list(gold_clusters=golds)

    document_store = create_file_doc_store(index_file, passages_file)
    predictions: List[QueryResult] = list()
    for query in query_examples.values():
        ret_passages: List[Passage] = document_store.get_passages_passages(query.id, top_k=top_k)
        query_obj = Query(query.__dict__)
        predictions.append(QueryResult(query_obj, ret_passages))

    to_print = print_measurements(predictions, golds_arranged, "retriever")
    join_result = "\n".join(to_print)
    print(join_result)


if __name__ == "__main__":
    SPLIT = "Dev"
    main()
