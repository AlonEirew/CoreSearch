import json
import logging
import pickle
from typing import List, Dict

from haystack.modeling.utils import set_all_seeds

from src.data_obj import Cluster, TrainExample, Passage, QueryResult
from src.index import elastic_index
from src.override_classes.reader.wec_reader import WECReader
from src.override_classes.retriever.wec_context_processor import WECContextProcessor
from src.override_classes.retriever.wec_dense import WECDensePassageRetriever
from src.pipeline.pipelines import QAPipeline, RetrievalOnlyPipeline
from src.utils import io_utils, measurments, data_utils, measure_squad
from src.utils.dpr_utils import create_file_doc_store
from src.utils.io_utils import write_json
from src.utils.measurments import precision, precision_squad


logger = logging.getLogger("run_haystack_pipeline")
logger.setLevel(logging.DEBUG)


SPLIT = "Test"


def main():
    # resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    # experiment can be one of {bm25, dpr, kenton}
    experiment = "DPR"
    prediction_head_str = experiment.lower()
    experiment_name = f"End_to_end_{experiment}_spc_special_1"
    # magnitude = all/cluster meaning if to use all queries (all) or just single clusters query (cluster)
    magnitude = "all"

    add_special_tokens = True

    # if experiment.lower() == "kenton":
    #     add_special_tokens = True
    # elif experiment.lower() in ["dpr", "bm25"]:
    #     add_special_tokens = False
    # elif query_method == "bm25":
    #     processor_type = WECBM25Processor
    #     add_special_tokens = False
    # else:
    #     raise ValueError(f"no such query method-{experiment}")

    ret_top_k = 500
    read_top_k = 50
    max_seq_len_query = 64
    max_seq_len_passage = 180
    batch_size = 24
    batch_size_qa = 24
    num_processes = 1

    set_all_seeds(seed=42)

    index_file = "file_indexes/" + SPLIT + "_Retriever_spanbert_5it1_top500.json"
    checkpoint_dir = "data/checkpoints/"
    query_encode = checkpoint_dir + "Retriever_SpanBERT_5it/1/query_encoder"
    passage_encode = checkpoint_dir + "Retriever_SpanBERT_5it/1/passage_encoder"
    #Reader - RoBERTa_base_DPR_special
    reader_model_file = checkpoint_dir + f"Reader-RoBERTa_base_{experiment}_special/1"

    gold_cluster_file = "data/resources/WEC-ES/clean/" + SPLIT + "_gold_clusters.json"
    queries_file = "data/resources/WEC-ES/train/" + SPLIT + "_queries.json"
    passages_file = "data/resources/WEC-ES/clean/" + SPLIT + "_all_passages.json"

    result_out_file = "results/" + SPLIT + "_" + experiment_name + ".txt"
    prediction_out_file = "results/" + SPLIT + "_" + experiment_name + "_predictions.pickle"

    golds: List[Cluster] = io_utils.read_gold_file(gold_cluster_file)
    query_examples: List[TrainExample] = io_utils.read_train_example_file(queries_file)

    if experiment.lower() in ["dpr", "kenton"]:
        document_store = create_file_doc_store(index_file, passages_file)
        logger.info("Total indexed documents to be searched=" + str(document_store.get_document_count()))
        passage_dict = document_store.passages
        retriever = WECDensePassageRetriever(document_store=document_store, query_embedding_model=query_encode,
                                             passage_embedding_model=passage_encode,
                                             infer_tokenizer_classes=True,
                                             max_seq_len_query=max_seq_len_query,
                                             max_seq_len_passage=max_seq_len_passage,
                                             batch_size=batch_size, use_gpu=True, embed_title=False,
                                             use_fast_tokenizers=False, processor_type=WECContextProcessor,
                                             add_special_tokens=add_special_tokens)

        pipeline = QAPipeline(document_store=document_store,
                              retriever=retriever,
                              reader=WECReader(model_name_or_path=reader_model_file, use_gpu=True,
                                               num_processes=num_processes,
                                               add_special_tokens=add_special_tokens,
                                               batch_size=batch_size_qa,
                                               replace_prediction_heads=True,
                                               prediction_head_str=prediction_head_str),
                              ret_topk=ret_top_k,
                              read_topk=read_top_k)
    elif experiment.lower() == "bm25":
        document_store, retriever = elastic_index.load_wec_elastic_bm25(SPLIT.lower())
        pipeline = RetrievalOnlyPipeline(document_store=document_store,
                                         retriever=retriever,
                                         ret_topk=ret_top_k)

        passage_examples: List[Passage] = io_utils.read_passages_file(passages_file)
        passage_dict: Dict[str, Passage] = {obj.id: obj for obj in passage_examples}
    else:
        raise ValueError(f"Experiment not supported!:{experiment}")

    query_examples = generate_query_text(passage_dict, query_examples, experiment, magnitude)

    logger.info("Running QA pipeline..")
    predict_and_eval(pipeline, golds, query_examples, experiment.lower(), result_out_file, prediction_out_file)


def predict_and_eval(pipeline, golds, query_examples, run_pipe_str, result_out_file, predict_out_file):
    predictions: List[QueryResult] = pipeline.run_end_to_end(query_examples=query_examples)
    golds_arranged = data_utils.clusters_to_ids_list(gold_clusters=golds)

    # assert len(predictions) == len(golds_arranged)
    print_results(predictions, golds_arranged, run_pipe_str, result_out_file)
    with open(predict_out_file, 'wb') as output:
        pickle.dump(predictions, output, pickle.HIGHEST_PROTOCOL)


def generate_query_text(passage_dict: Dict[str, Passage], query_examples: List[TrainExample], query_method: str = None, magnitude="all"):
    used_clusters = set()
    ret_queries = list()
    logger.info("Using query style-" + query_method)
    for query in query_examples:
        if magnitude == "cluster" and query.goldChain in used_clusters:
            continue

        used_clusters.add(query.goldChain)
        ret_queries.append(query)
        if query_method:
            if query_method == "bm25":
                query.context = query.bm25_query.split(" ")
            # elif query_method == "ment_only":
            #     query.context = query.mention
            # elif query_method == "with_bounds":
            #     pass
            # elif query_method == "full_ctx":
            else:
                pass

        if passage_dict:
            for pass_id in query.positive_examples:
                query.answers.add(" ".join(passage_dict[pass_id].mention))

    logger.info(f"Total generated queries-{len(ret_queries)}")
    return ret_queries


def print_results(predictions, golds_arranged, run_pipe_str, result_out_file):
    # Print retriever evaluation matrices
    to_print = print_measurements(predictions, golds_arranged, run_pipe_str)

    to_print.append("#################################################################################################")
    delimiter = "#################"
    for query_result in predictions:
        query_coref_id = str(query_result.query.goldChain)
        to_print.append(delimiter + " " + query_coref_id + " " + delimiter)
        query_context = " ".join(query_result.query.context)
        to_print.append("QUERY_ANSWERS=" + str(query_result.query.answers))
        to_print.append("QUERY_CONTEXT=" + query_context)
        to_print.append("QUERY_ID=" + query_result.query.id)
        for r_ind, result in enumerate(query_result.results[:10]):
            to_print.append("\tRESULT-" + str(result.id) + ":")
            to_print.append("\t\tCOREF_ID=" + str(result.goldChain))
            result_context = result.context
            result_answer = "NA_RETRIEVER"
            result_mention = result.mention
            if result.answer:
                result_answer = result.answer
            to_print.append("\t\tANSWER=" + str(result_answer))
            to_print.append("\t\tGOLD_MENTION=" + str(result_mention))
            to_print.append("\t\tSCORE=" + str(result.score))
            to_print.append("\t\tCONTEXT=" + str(result_context))

    join_result = "\n".join(to_print)
    # print(join_result)

    logger.info("Saving report to-" + result_out_file)
    with open(result_out_file, 'w') as f:
        f.write(join_result)


def print_measurements(predictions, golds_arranged, run_pipe_str=None):
    to_print = list()
    if run_pipe_str and run_pipe_str in ["dpr", "kenton"]:
        to_print.append("---------- Evaluation of Reader Model ------------")
        precision_method = precision_squad
        # Print the squad evaluation matrices
        to_print.append(json.dumps(measure_squad.eval_qa(predictions)))
        to_print.append("MRR@10=" + str(measurments.mean_reciprocal_rank(
            predictions=predictions, golds=golds_arranged, topk=10, method=run_pipe_str)))
        to_print.append("mAP@5=" + str(measurments.mean_average_precision(
            predictions=predictions, golds=golds_arranged, precision_method=precision_method, topk=5)))
        to_print.append("mAP@10=" + str(measurments.mean_average_precision(
            predictions=predictions, golds=golds_arranged, precision_method=precision_method, topk=10)))
        to_print.append("mAP@25=" + str(measurments.mean_average_precision(
            predictions=predictions, golds=golds_arranged, precision_method=precision_method, topk=25)))
        to_print.append("mAP@50=" + str(measurments.mean_average_precision(
            predictions=predictions, golds=golds_arranged, precision_method=precision_method, topk=50)))
    to_print.append("---------- Evaluation of Retriever Model ------------")
    precision_method = precision
    to_print.append("MRR@10=" + str(measurments.mean_reciprocal_rank(
        predictions=predictions, golds=golds_arranged, topk=10, method=run_pipe_str)))
    to_print.append("mAP@5=" + str(measurments.mean_average_precision(
        predictions=predictions, golds=golds_arranged, precision_method=precision_method, topk=5)))
    to_print.append("mAP@10=" + str(measurments.mean_average_precision(
        predictions=predictions, golds=golds_arranged, precision_method=precision_method, topk=10)))
    to_print.append("mAP@25=" + str(measurments.mean_average_precision(
        predictions=predictions, golds=golds_arranged, precision_method=precision_method, topk=25)))
    to_print.append("mAP@50=" + str(measurments.mean_average_precision(
        predictions=predictions, golds=golds_arranged, precision_method=precision_method, topk=50)))
    to_print.append("mAP@100=" + str(measurments.mean_average_precision(
        predictions=predictions, golds=golds_arranged, precision_method=precision_method, topk=100)))

    to_print.append("Recall@5=" + str(measurments.recall(
        predictions=predictions, golds=golds_arranged, topk=5)))
    to_print.append("Recall@10=" + str(measurments.recall(
        predictions=predictions, golds=golds_arranged, topk=10)))
    to_print.append("Recall@25=" + str(measurments.recall(
        predictions=predictions, golds=golds_arranged, topk=25)))
    to_print.append("Recall@50=" + str(measurments.recall(
        predictions=predictions, golds=golds_arranged, topk=50)))
    to_print.append("Recall@100=" + str(measurments.recall(
        predictions=predictions, golds=golds_arranged, topk=100)))
    to_print.append("Recall@200=" + str(measurments.recall(
        predictions=predictions, golds=golds_arranged, topk=200)))
    to_print.append("Recall@300=" + str(measurments.recall(
        predictions=predictions, golds=golds_arranged, topk=300)))
    to_print.append("Recall@400=" + str(measurments.recall(
        predictions=predictions, golds=golds_arranged, topk=400)))
    to_print.append("Recall@500=" + str(measurments.recall(
        predictions=predictions, golds=golds_arranged, topk=500)))
    return to_print


if __name__ == '__main__':
    main()
