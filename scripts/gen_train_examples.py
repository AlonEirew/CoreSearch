"""
This script generate the training files which include negative samples collected by the top BM25 negative results, similarly to what was done in DPR.
Prerequisite for running this script is generating the Elasticsearch index using elastic_index.py script
"""
import json
from typing import List, Tuple, Set

from tqdm import tqdm

from src.data_obj import Query, Cluster, QueryResult, TrainExample, Passage
from src.index import elastic_index
from src.pipeline.pipelines import RetrievalOnlyPipeline
from src.utils import io_utils, data_utils
from src.utils.nlp_utils import NLPUtils


def create_train_examples(query_results, golds) -> Tuple[List[TrainExample], Set[str]]:
    assert query_results
    assert golds
    train_examples = list()
    all_pass_ids = set()
    predictions_arranged = data_utils.query_results_to_ids_list(query_results)
    golds_arranged = data_utils.clusters_to_ids_list(gold_clusters=golds)
    for query_res in query_results:
        predicted_list: List[str] = predictions_arranged[query_res.query.id]
        gold_list: List[str] = golds_arranged[query_res.query.id]
        neg_ids = list()
        for rid in predicted_list:
            all_pass_ids.update(gold_list)
            if rid not in gold_list:
                neg_ids.append(rid)
                all_pass_ids.add(rid)

        json_obj = query_res.query.__dict__
        json_obj["positive_examples"] = gold_list
        json_obj["negative_examples"] = neg_ids
        json_obj["bm25_query"] = query_res.searched_query

        train_examples.append(TrainExample(json_obj))

    return train_examples, all_pass_ids


def extract_bm25_query_and_run(nlp, pipeline, query_examples) -> List[QueryResult]:
    assert query_examples
    query_results = list()
    for query in query_examples:
        query_mention = " ".join(query.mention)
        query_ners: List[Tuple[str, str, str, str]] = nlp.extract_ner_spans(" ".join(query.context))
        ner_text = set([ner[1] for ner in query_ners])
        query_text = query_mention + " . " + " ".join(ner_text)
        result = pipeline.run_pipeline(query_text)
        query_res = pipeline.extract_results(query, result)
        query_res.searched_query = query_text
        query_results.append(query_res)
    return query_results


def filter_only_used_passages(passages_file, all_pass_ids) -> List[Passage]:
    all_passages: List[Passage] = io_utils.read_passages_file(passages_file)
    filtered_passages: List[Passage] = list()
    for passage in tqdm(all_passages, "Passage Filtering"):
        if passage.id in all_pass_ids:
            filtered_passages.append(passage)

    print(SPLIT + " passages count *before* filtering=" + str(len(all_passages)))
    print(SPLIT + " passages count *after* filtering=" + str(len(filtered_passages)))
    return filtered_passages


def main():
    passages_file = "resources/WEC-ES/" + SPLIT + "_passages.json"
    queries_file = "resources/WEC-ES/" + SPLIT + "_queries.json"
    gold_file = "resources/WEC-ES/" + SPLIT + "_gold_clusters.json"

    train_exml_out_file = "resources/train/" + SPLIT + "_training_queries.json"
    train_filtered_pass_out = "resources/train/" + SPLIT + "_training_passages.json"

    topk = 20

    golds: List[Cluster] = io_utils.read_gold_file(gold_file)
    query_examples: List[Query] = io_utils.read_query_file(queries_file)
    document_store, retriever = elastic_index.load_elastic_bm25(SPLIT.lower())
    pipeline = RetrievalOnlyPipeline(document_store=document_store,
                                     retriever=retriever,
                                     ret_topk=topk)
    nlp = NLPUtils()
    query_results = extract_bm25_query_and_run(nlp, pipeline, query_examples)
    train_examples, all_pass_ids = create_train_examples(query_results, golds)
    filtered_passages = filter_only_used_passages(passages_file, all_pass_ids)

    print(SPLIT + " train queries=" + str(len(train_examples)))
    print("Writing-" + train_exml_out_file)
    with open(train_exml_out_file, 'w', encoding='utf-8') as train_exmpl_os:
        json.dump(train_examples, train_exmpl_os, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)

    print("Writing-" + train_filtered_pass_out)
    with open(train_filtered_pass_out, 'w', encoding='utf-8') as filtered_pass_os:
        json.dump(filtered_passages, filtered_pass_os, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    SPLIT = "Train"
    main()
