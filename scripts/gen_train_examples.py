"""
This script generate the training files which include negative samples collected by the top BM25 negative results,
this is similar to what was done in DPR.
Prerequisite for running this script is generating the Elasticsearch index using elastic_index.py script
"""
import json
from typing import List, Tuple, Set

import spacy
from tqdm import tqdm

from src.data_obj import Query, Cluster, QueryResult, TrainExample, Passage
from src.index import elastic_index
from src.pipeline.pipelines import RetrievalOnlyPipeline
from src.utils import io_utils, data_utils


def create_train_examples(query_results, golds, negative_sample) -> Tuple[List[TrainExample], Set[str]]:
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
        json_obj["negative_examples"] = neg_ids[:negative_sample]
        json_obj["bm25_query"] = query_res.searched_query

        example = TrainExample(json_obj)
        # No need and because json throws exception (set())
        example.answers = None
        train_examples.append(example)

    return train_examples, all_pass_ids


def extract_bm25_query_and_run(pipeline, query_examples) -> List[QueryResult]:
    assert query_examples
    query_results = list()
    spacy_parser = spacy.load('en_core_web_trf')
    for query in tqdm(query_examples, desc="Running BM25"):
        query_mention = " ".join(query.mention)
        query_ners: List[Tuple[str, str, str, str, str]] = extract_ner_spans(spacy_parser, " ".join(query.context))
        ner_text = set([ner[0] for ner in query_ners])
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


def extract_ner_spans(spacy_parser, context: str) -> List[Tuple[str, str, str, str, str]]:
    doc = spacy_parser(context)
    ents = [(e.text, e.lemma_, e.start_char, e.end_char, e.label_) for e in doc.ents]
    return ents


def main():
    passages_file = "data/resources/WEC-ES/clean/" + SPLIT + "_all_passages.json"
    queries_file = "data/resources/WEC-ES/clean/" + SPLIT + "_queries.json"
    gold_file = "data/resources/WEC-ES/clean/" + SPLIT + "_gold_clusters.json"

    train_exml_out_file = "data/resources/WEC-ES/train/" + SPLIT + "_queries.json"
    train_filtered_pass_out = "data/resources/WEC-ES/train/" + SPLIT + "_passages.json"

    topk = 100
    negative_sample = 25

    golds: List[Cluster] = io_utils.read_gold_file(gold_file)
    query_examples: List[Query] = io_utils.read_query_file(queries_file)
    document_store, retriever = elastic_index.load_elastic_bm25(SPLIT.lower())
    pipeline = RetrievalOnlyPipeline(document_store=document_store,
                                     retriever=retriever,
                                     ret_topk=topk)
    query_results = extract_bm25_query_and_run(pipeline, query_examples)
    train_examples, all_pass_ids = create_train_examples(query_results, golds, negative_sample)
    filtered_passages = filter_only_used_passages(passages_file, all_pass_ids)

    print(SPLIT + " train queries=" + str(len(train_examples)))
    print("Writing-" + train_exml_out_file)
    with open(train_exml_out_file, 'w', encoding='utf-8') as train_exmpl_os:
        json.dump(train_examples, train_exmpl_os, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)

    print("Writing-" + train_filtered_pass_out)
    with open(train_filtered_pass_out, 'w', encoding='utf-8') as filtered_pass_os:
        json.dump(filtered_passages, filtered_pass_os, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)

    print("Done!")


if __name__ == '__main__':
    SPLIT = "Train"
    main()
