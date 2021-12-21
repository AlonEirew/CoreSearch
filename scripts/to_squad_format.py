import json
import os
import re
from typing import Dict, List

from src.data_obj import TrainExample, Cluster
from src.utils import io_utils


def main():
    train_queries: Dict[str, TrainExample] = TrainExample.list_to_map(io_utils.read_train_example_file("resources/train/" + SPLIT + "_training_queries.json"))
    clusters: List[Cluster] = io_utils.read_gold_file("resources/WEC-ES/" + SPLIT + "_gold_clusters.json")

    squad_out = "resources/squad/" + SPLIT + "_squad_format.json"

    assert train_queries
    assert clusters

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
            qas_list.extend(create_qas_list(pos_query_id_list, train_queries, query_ans, context, is_impossible=False))
            # qas_list.extend(create_qas_list(imp_query_id_list, train_queries, query_ans, context, is_impossible=True))

            paragraph["qas"] = qas_list
            paragraph["context"] = context
            paragraphs.append(paragraph)

        data.append(data_obj)

    os.makedirs(os.path.dirname(squad_out), exist_ok=True)
    with open(squad_out, 'w', encoding='utf-8') as train_exmpl_os:
        json.dump(squad_examples, train_exmpl_os, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)


def create_qas_list(query_id_list, train_queries, query_ans, context, is_impossible):
    qas_list = list()
    for query_id in query_id_list:
        query = train_queries[query_id]
        qas_obj = dict()
        answers = list()
        qas_obj["question"] = query.bm25_query
        qas_obj["id"] = query.id
        qas_obj["answers"] = answers
        qas_obj["is_impossible"] = is_impossible
        if not is_impossible:
            start_idxs = [m.start() for m in re.finditer(query_ans, context)]
            for index in start_idxs:
                ans_obj = {"text": query_ans, "answer_start": index}
                answers.append(ans_obj)

        qas_list.append(qas_obj)
    return qas_list


if __name__ == '__main__':
    SPLIT = "Train"
    main()
    print("Done generating for " + SPLIT)
