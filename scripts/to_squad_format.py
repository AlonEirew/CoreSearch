import json
import os
from typing import Dict, List

from src.data_obj import TrainExample, Cluster, Passage
from src.utils import io_utils
from src.utils.io_utils import load_json_file, read_passages_file

SPLIT = "Test"


def main():
    train_queries: Dict[str, TrainExample] = TrainExample.list_to_map(io_utils.read_train_example_file(
        "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/train/min_span/" + SPLIT + "_queries_min_span.json"))
    clusters: List[Cluster] = io_utils.read_gold_file("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/min_span/" + SPLIT + "_gold_clusters.json")
    passages_file = "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/min_span/" + SPLIT + "_passages_min_span.json"
    passage_dict: Dict[str, Passage] = {obj.id: obj for obj in read_passages_file(passages_file)}

    retriver_file = "file_indexes/" + SPLIT + "_Retriever_spanbert_5it0_top500.json"
    retriever_results = load_json_file(retriver_file)

    assert train_queries
    assert clusters
    query_style = "context"
    num_of_positives = 1
    num_of_negatives = 23

    if query_style == "bm25":
        squad_out = "data/resources/squad/bm25/" + SPLIT + "_squad_format.json"
    elif query_style == "context":
        squad_out = "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/squad/min_span/" + SPLIT + "_squad_format_1pos_23neg.json"
    else:
        raise ValueError(f"Not a supported query_style-{query_style}")

    squad_examples = dict()
    data = list()
    squad_examples["data"] = data

    discarded_queries = 0
    all_contexts = dict()
    for clust in clusters:
        for ment_id in clust.mention_ids:
            if ment_id in train_queries:
                query_example = train_queries[ment_id]
                pos_exampl_id_list = list()
                neg_exampl_id_list = list()
                # Selecting query top candidates from retrieved (DPR) results passages
                for res in retriever_results[ment_id]:
                    if res["pass_id"] in clust.mention_ids:
                        pos_exampl_id_list.append(res["pass_id"])
                    else:
                        neg_exampl_id_list.append(res["pass_id"])

                pos_exampl_id_list = pos_exampl_id_list[:num_of_positives]
                neg_exampl_id_list = neg_exampl_id_list[:num_of_negatives]
                if len(pos_exampl_id_list) == 0:
                    discarded_queries += 1
                    continue

                assert_as_required = 0
                for ctx_id in pos_exampl_id_list:
                    if ctx_id not in all_contexts:
                        all_contexts[ctx_id] = list()
                    all_contexts[ctx_id].append(create_qas_obj(query_example, False, query_style, passage_dict[ctx_id]))
                    assert_as_required += 1

                for ctx_id in neg_exampl_id_list:
                    if ctx_id not in all_contexts:
                        all_contexts[ctx_id] = list()
                    all_contexts[ctx_id].append(create_qas_obj(query_example, True, query_style))
                    assert_as_required += 1

                assert assert_as_required == num_of_positives + num_of_negatives

    print(f"Discarded queries={discarded_queries}")

    for ctx_id, qas_list in all_contexts.items():
        paragraphs = list()
        data_obj = dict()
        data_obj['paragraphs'] = paragraphs

        paragraph = dict()
        paragraph['title'] = int(passage_dict[ctx_id].goldChain)
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
        # start_idxs = [m.start() for m in re.finditer(re.escape(query_ans), " ".join(passage.context))]
        start_idx = len(" ".join(passage.context[:passage.startIndex]))
        if start_idx > 0:
            # if start_index > 0 we need to add the space to get to the start of the answer
            start_idx += 1

        ans_obj = {"text": query_ans, "answer_start": start_idx,
                   "ment_start": passage.startIndex, "ment_end": passage.endIndex,
                   "ment_text": passage.mention}
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
    qas_obj["ment_start"] = query_example.startIndex
    qas_obj["ment_end"] = query_example.endIndex
    qas_obj["query_mention"] = query_example.mention
    qas_obj["query_coref_link"] = int(query_example.goldChain)
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
    main()
    print("Done generating for " + SPLIT)
