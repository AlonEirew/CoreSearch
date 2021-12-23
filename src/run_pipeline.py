import json
from typing import List

from haystack.reader import FARMReader

from src.data_obj import Cluster, TrainExample, Passage, QueryResult
from src.index import faiss_index, elastic_index
from src.pipeline.pipelines import QAPipeline, RetrievalOnlyPipeline
from src.utils import io_utils, measurments, data_utils, eval_squad


def main():
    # index_type = "elastic_bm25"
    index_type = "faiss_dpr"
    # run_pipe_str = "retriever"
    run_pipe_str = "qa"
    es_index = SPLIT.lower()
    lanuguage_model = "spanbert"

    faiss_index_file = "weces_index_for_" + lanuguage_model + "_dpr/wec_" + es_index + "_index.faiss"
    sql_url = "sqlite:///weces_index_for_" + lanuguage_model + "_dpr/weces_" + es_index + ".db"

    retriever_model = "data/checkpoints/dpr_" + lanuguage_model + "_best"
    reader_model = "data/checkpoints/squad_spanbert_2it"

    gold_cluster_file = "data/resources/WEC-ES/" + SPLIT + "_gold_clusters.json"
    queries_file = "data/resources/train/" + SPLIT + "_training_queries.json"
    passages_file = "data/resources/train/" + SPLIT + "_training_passages.json"

    golds: List[Cluster] = io_utils.read_gold_file(gold_cluster_file)
    # query_examples: List[Query] = io_utils.read_query_file("resources/WEC-ES/" + SPLIT + "_queries.json")
    query_examples: List[TrainExample] = io_utils.read_train_example_file(queries_file)
    passage_examples: List[Passage] = io_utils.read_passages_file(passages_file)

    passage_dict = {obj.id: obj for obj in passage_examples}
    for query in query_examples:
        query.context = query.bm25_query.split(" ")
        for pass_id in query.positive_examples:
            query.answers.add(" ".join(passage_dict[pass_id].mention))

    if index_type == "faiss_dpr":
        document_store, retriever = faiss_index.load_faiss_dpr(faiss_index_file, sql_url, retriever_model)
    elif index_type == "elastic_bm25":
        document_store, retriever = elastic_index.load_elastic_bm25(es_index)
    else:
        raise TypeError

    print(index_type + " Document store and retriever created..")
    print("Total indexed documents to be searched=" + str(document_store.get_document_count()))

    if run_pipe_str == "qa":
        pipeline = QAPipeline(document_store=document_store,
                              retriever=retriever,
                              # reader=FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True),
                              reader=FARMReader(model_name_or_path=reader_model, use_gpu=True),
                              ret_topk=100,
                              read_topk=10)
    elif run_pipe_str == "retriever":
        pipeline = RetrievalOnlyPipeline(document_store=document_store,
                                         retriever=retriever,
                                         ret_topk=150)
    else:
        raise TypeError

    print("Running " + run_pipe_str + " pipeline..")
    predict_and_eval(pipeline, golds, query_examples, run_pipe_str)


def print_results(predictions: List[QueryResult]):
    to_print = list()
    delimiter = "#################"
    for query_result in predictions:
        query_coref_id = str(query_result.query.goldChain)
        to_print.append(delimiter + " " + query_coref_id + " " + delimiter)
        query_context = " ".join(query_result.query.context)
        to_print.append("QUERY_ANSWERS=" + str(query_result.query.answers))
        to_print.append("QUERY_CONTEXT=" + query_context)
        for r_ind, result in enumerate(query_result.results[:5]):
            to_print.append("\tRESULT" + str(r_ind) + ":")
            to_print.append("\t\tCOREF_ID=" + str(result.goldChain))
            result_context = result.context
            result_answer = "NA_RETRIEVER"
            result_mention = result.mention
            if result.answer:
                result_answer = result.answer
            to_print.append("\t\tANSWER=" + str(result_answer))
            to_print.append("\t\tGOLD_MENTION=" + result_mention)
            to_print.append("\t\tCONTEXT=" + str(result_context))

    print("\n".join(to_print))


def predict_and_eval(pipeline, golds, query_examples, run_pipe_str):
    predictions: List[QueryResult] = pipeline.run_end_to_end(query_examples=query_examples)
    predictions_arranged = data_utils.query_results_to_ids_list(predictions)
    golds_arranged = data_utils.clusters_to_ids_list(gold_clusters=golds)

    # assert len(predictions) == len(golds_arranged)
    # print("HIT@10=" + str(measurments.hit_rate(predictions=predictions_arranged, golds=golds_arranged, topk=10)))

    if run_pipe_str == "qa":
        # Print the squad evaluation matrices
        print(json.dumps(eval_squad.eval_qa(predictions)))

    # Print retriever evaluation matrices
    print("MRR@10=" + str(measurments.mean_reciprocal_rank(predictions=predictions_arranged, golds=golds_arranged, topk=10)))
    print("RECALL@10=" + str(measurments.recall(predictions=predictions_arranged, golds=golds_arranged, topk=10)))
    print("RECALL@50=" + str(measurments.recall(predictions=predictions_arranged, golds=golds_arranged, topk=50)))
    print("RECALL@100=" + str(measurments.recall(predictions=predictions_arranged, golds=golds_arranged, topk=100)))
    print("####################################################################################################")
    print_results(predictions)


if __name__ == '__main__':
    SPLIT = "Dev"
    main()
