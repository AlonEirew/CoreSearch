import json
import os
import re
from typing import Dict, List

from src.data_obj import TrainExample, Cluster, Passage
from src.utils import io_utils
from src.utils.io_utils import load_json_file, read_passages_file


def main():
    train_queries: Dict[str, TrainExample] = TrainExample.list_to_map(io_utils.read_train_example_file(
        "data/resources/train/" + SPLIT + "_training_queries.json"))
    clusters: List[Cluster] = io_utils.read_gold_file("data/resources/WEC-ES/" + SPLIT + "_gold_clusters.json")
    passages_file = "data/resources/WEC-ES/" + SPLIT + "_all_passages.json"
    result_file = "file_indexes/" + SPLIT.lower() + "_spanbert_hidden_cls_spatial_ctx_2it_top500.json"

    passage_dict: Dict[str, Passage] = {obj.id: obj for obj in read_passages_file(passages_file)}
    results_dict = load_json_file(result_file)

    assert train_queries
    assert clusters
    query_style = "context"

    if query_style == "bm25":
        squad_out = "data/resources/squad/bm25/" + SPLIT + "_squad_format.json"
    elif query_style == "context":
        squad_out = "data/resources/squad/context/" + SPLIT + "_squad_format_1pos.json"
    else:
        raise ValueError(f"Not a supported query_style-{query_style}")

    squad_examples = dict()
    data = list()
    squad_examples["data"] = data

    all_contexts = dict()
    for clust in clusters:
        for ment_id in clust.mention_ids:
            query_example = train_queries[ment_id]
            pos_exampl_id_list = list()
            neg_exampl_id_list = list()
            for res in results_dict[ment_id]:
                if res["pass_id"] in clust.mention_ids:
                    pos_exampl_id_list.append(res["pass_id"])
                else:
                    neg_exampl_id_list.append(res["pass_id"])

            pos_exampl_id_list = pos_exampl_id_list[:1]
            neg_exampl_id_list = neg_exampl_id_list[:24]
            for ctx_id in pos_exampl_id_list:
                if ctx_id not in all_contexts:
                    all_contexts[ctx_id] = list()
                all_contexts[ctx_id].append(create_qas_obj(query_example, False, query_style, passage_dict[ctx_id]))

            for ctx_id in neg_exampl_id_list:
                if ctx_id not in all_contexts:
                    all_contexts[ctx_id] = list()
                all_contexts[ctx_id].append(create_qas_obj(query_example, True, query_style))

    for ctx_id, qas_list in all_contexts.items():
        paragraphs = list()
        data_obj = dict()
        data_obj['title'] = passage_dict[ctx_id].goldChain
        data_obj['paragraphs'] = paragraphs

        paragraph = dict()
        paragraph["qas"] = qas_list
        paragraph["context"] = " ".join(passage_dict[ctx_id].context)
        paragraphs.append(paragraph)

        data.append(data_obj)

    os.makedirs(os.path.dirname(squad_out), exist_ok=True)
    with open(squad_out, 'w', encoding='utf-8') as train_exmpl_os:
        json.dump(squad_examples, train_exmpl_os, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)


def create_qas_obj(query_example, is_impossible, query_style, passage=None):
    qas_obj = dict()
    answers = list()
    if not is_impossible:
        query_ans = " ".join(passage.mention)
        start_idxs = [m.start() for m in re.finditer(re.escape(query_ans), " ".join(passage.context))]
        for index in start_idxs:
            ans_obj = {"text": query_ans, "answer_start": index,
                       "ment_start": passage.startIndex, "ment_end": passage.endIndex, "ment_text": passage.mention}
            answers.append(ans_obj)
    else:
        ans_obj = {"text": "", "answer_start": 0,
                   "ment_start": 0, "ment_end": 0,
                   "ment_text": ["NA"]}
        answers.append(ans_obj)

    if query_style == "bm25":
        qas_obj["question"] = query_example.bm25_query
    else:
        qas_obj["question"] = " ".join(query_example.context)
    qas_obj["id"] = query_example.id
    qas_obj["answers"] = answers
    qas_obj["is_impossible"] = is_impossible
    return qas_obj


def create_qas_list(query_id_list, train_queries, query_style):
    qas_list = list()
    for query_id in query_id_list:
        qas_obj = create_qas_obj(train_queries[query_id], False, query_style)
        qas_list.append(qas_obj)
    return qas_list


if __name__ == '__main__':
    SPLIT = "Dev"
    main()
    print("Done generating for " + SPLIT)
