import json
from typing import List

from haystack.nodes import FARMReader

from src.data_obj import Cluster, TrainExample, Passage, QueryResult
from src.index import elastic_index
from src.pipeline.pipelines import QAPipeline, RetrievalOnlyPipeline
from src.utils import io_utils, measurments, data_utils, eval_squad, dpr_utils


def main():
    # index_type = "elastic_bm25"
    index_type = "faiss_dpr"
    run_pipe_str = "retriever"
    # run_pipe_str = "qa"
    es_index = SPLIT.lower()
    language_model = "multiset"

    checkpoint_dir = "data/checkpoints/"
    faiss_index_prefix = "weces_index_" + language_model + "/weces_" + es_index + "_index"
    faiss_index_file = faiss_index_prefix + ".faiss"
    faiss_config_file = faiss_index_prefix + ".json"

    query_encode = "facebook/dpr-question_encoder-multiset-base"
    passage_encode = "facebook/dpr-ctx_encoder-multiset-base"
    retriever_model_file = None # dpr_" + language_model + "_best"
    reader_model_file = None # "squad_roberta_best"

    gold_cluster_file = "data/resources/WEC-ES/" + SPLIT + "_gold_clusters.json"
    queries_file = "data/resources/train/" + SPLIT + "_training_queries.json"
    passages_file = "data/resources/train/" + SPLIT + "_training_passages.json"
    result_out_file = "results/" + es_index + "_" + language_model + "_" + \
                      str(retriever_model_file) + "_" + str(reader_model_file) + ".txt"

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
        if retriever_model_file:
            retriever_model = checkpoint_dir + retriever_model_file
            document_store, retriever = dpr_utils.load_faiss_dpr(faiss_index_file, faiss_config_file, retriever_model)
        else:
            document_store = dpr_utils.load_faiss_doc_store(faiss_index_file, faiss_config_file)
            retriever = dpr_utils.create_default_dpr(document_store, query_encode, passage_encode)
    elif index_type == "elastic_bm25":
        document_store, retriever = elastic_index.load_elastic_bm25(es_index)
    else:
        raise TypeError

    print(index_type + " Document store and retriever created..")
    print("Total indexed documents to be searched=" + str(document_store.get_document_count()))

    if run_pipe_str == "qa":
        reader_model = checkpoint_dir + reader_model_file
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
    predict_and_eval(pipeline, golds, query_examples, run_pipe_str, result_out_file)


def print_results(predictions_arranged, predictions, golds_arranged, result_out_file):
    # Print retriever evaluation matrices
    to_print = list()
    to_print.append("MRR@10=" + str(measurments.mean_reciprocal_rank(predictions=predictions_arranged, golds=golds_arranged, topk=10)))
    to_print.append("RECALL@10=" + str(measurments.recall(predictions=predictions_arranged, golds=golds_arranged, topk=10)))
    to_print.append("RECALL@50=" + str(measurments.recall(predictions=predictions_arranged, golds=golds_arranged, topk=50)))
    to_print.append("RECALL@100=" + str(measurments.recall(predictions=predictions_arranged, golds=golds_arranged, topk=100)))
    to_print.append("####################################################################################################")
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

    join_result = "\n".join(to_print)
    print(join_result)
    with open(result_out_file, 'w') as f:
        f.write(join_result)


def predict_and_eval(pipeline, golds, query_examples, run_pipe_str, result_out_file):
    predictions: List[QueryResult] = pipeline.run_end_to_end(query_examples=query_examples)
    predictions_arranged = data_utils.query_results_to_ids_list(predictions)
    golds_arranged = data_utils.clusters_to_ids_list(gold_clusters=golds)

    # assert len(predictions) == len(golds_arranged)
    # print("HIT@10=" + str(measurments.hit_rate(predictions=predictions_arranged, golds=golds_arranged, topk=10)))

    if run_pipe_str == "qa":
        # Print the squad evaluation matrices
        print(json.dumps(eval_squad.eval_qa(predictions)))

    print_results(predictions_arranged, predictions, golds_arranged, result_out_file)


if __name__ == '__main__':
    SPLIT = "Dev"
    main()
