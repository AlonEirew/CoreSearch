import json
import os
import re
from typing import Dict, List

from src.data_obj import TrainExample, Cluster
from src.utils import io_utils


def main():
    train_queries: Dict[str, TrainExample] = TrainExample.list_to_map(io_utils.read_train_example_file(
        "data/resources/train/" + SPLIT + "_training_queries.json"))
    clusters: List[Cluster] = io_utils.read_gold_file("data/resources/WEC-ES/" + SPLIT + "_gold_clusters.json")

    assert train_queries
    assert clusters
    query_style = "context"

    if query_style == "bm25":
        squad_out = "data/resources/squad/bm25/" + SPLIT + "_squad_format.json"
    else:
        squad_out = "data/resources/squad/context/" + SPLIT + "_squad_format.json"

    squad_examples = dict()
    data = list()
    squad_examples["data"] = data

    for clust in clusters:
        data_obj = dict()
        paragraphs = list()
        data_obj['title'] = "_".join(clust.cluster_title.split(" "))
        data_obj['paragraphs'] = paragraphs
        for ment_id in clust.mention_ids:
            train_query = train_queries[ment_id]
            context = " ".join(train_query.context)
            pos_query_id_list = train_query.positive_examples
            # imp_query_id_list = list(filter(lambda x: not x.startswith("NEG"), train_query.negative_examples))
            paragraph = dict()
            qas_list = list()
            query_ans = " ".join(train_query.mention)
            # Will generate for each cluster mention a quesion related to current context
            qas_list.extend(create_qas_list(pos_query_id_list, train_queries, query_ans, train_query, query_style))
            # qas_list.extend(create_qas_list(imp_query_id_list, train_queries, query_ans, context, is_impossible=True))

            paragraph["qas"] = qas_list
            paragraph["context"] = context
            paragraphs.append(paragraph)

        data.append(data_obj)

    os.makedirs(os.path.dirname(squad_out), exist_ok=True)
    with open(squad_out, 'w', encoding='utf-8') as train_exmpl_os:
        json.dump(squad_examples, train_exmpl_os, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)


def create_qas_list(query_id_list, train_queries, query_ans, context_ment, query_style):
    qas_list = list()
    for query_id in query_id_list:
        query = train_queries[query_id]
        qas_obj = dict()
        answers = list()
        if query_style == "bm25":
            qas_obj["question"] = query.bm25_query
        else:
            qas_obj["question"] = " ".join(query.context)
        qas_obj["id"] = query.id
        qas_obj["answers"] = answers
        qas_obj["is_impossible"] = False
        start_idxs = [m.start() for m in re.finditer(re.escape(query_ans), " ".join(context_ment.context))]
        for index in start_idxs:
            ans_obj = {"text": query_ans, "answer_start": index,
                       "ment_start": query.startIndex, "ment_end": query.endIndex, "ment_text": query.mention}
            answers.append(ans_obj)

        qas_list.append(qas_obj)
    return qas_list


if __name__ == '__main__':
    SPLIT = "Dev"
    main()
    print("Done generating for " + SPLIT)
