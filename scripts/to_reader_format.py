"""
Script for generating the DPR format for training the retriever model.
"""

import json
from typing import Dict

from tqdm import tqdm

from src.data_obj import TrainExample, Passage, DPRContext, DPRExample
from src.utils import io_utils
from src.utils.io_utils import read_passages_file, load_json_file


def main():
    queries: Dict[str, TrainExample] = TrainExample.list_to_map(io_utils.read_train_example_file(
        "data/resources/WEC-ES/train/" + SPLIT + "_queries.json"))
    passages_file = "data/resources/WEC-ES/clean/" + SPLIT + "_all_passages.json"
    passage_dict: Dict[str, Passage] = {obj.id: obj for obj in read_passages_file(passages_file)}

    retriver_file = "file_indexes/" + SPLIT + "_Baseline4_spanbert_2it_top500.json"
    retriever_results = load_json_file(retriver_file)

    # query_style_bm25 if true will replace the query.context with the bm25 query
    # permute_all_positive = True
    num_of_negatives = 23
    num_of_positives = 1

    dpr_out = "data/resources/squad/reader/" + SPLIT + ".json"

    assert queries
    assert passage_dict
    discarded_queries = 0
    dpr_examples = list()
    for qid, query in tqdm(queries.items(), "Converting"):
        pos_exampl_id_list = list()
        neg_exampl_id_list = list()
        # Selecting query top candidates from retrieved (DPR) results passages
        for res in retriever_results[qid]:
            if res["pass_id"] == query.id:
                pos_exampl_id_list.append(res["pass_id"])
            else:
                neg_exampl_id_list.append(res["pass_id"])

        pos_exampl_id_list = pos_exampl_id_list[:num_of_positives]
        neg_exampl_id_list = neg_exampl_id_list[:num_of_negatives]
        if len(pos_exampl_id_list) == 0:
            discarded_queries += 1
            continue

        dpr_question = " ".join(query.context)

        dpr_answers = set()
        for pos_pass_id in pos_exampl_id_list:
            pos_passage = passage_dict[pos_pass_id]
            dpr_answers.add(" ".join(pos_passage.mention))
            dpr_pos_ctxs.append(DPRContext("NA", " ".join(pos_passage.context), 0, 0, pos_pass_id))
        for neg_pass_id in negatives:
            neg_passage = passage_dict[neg_pass_id]
            dpr_neg_ctxs.append(DPRContext("NA", " ".join(neg_passage.context), 0, 0, neg_pass_id))

        # if permute_all_positive:
        for dpr_pos in dpr_pos_ctxs:
            dpr_examples.append(DPRExample(dataset="WEC",
                                           question=dpr_question,
                                           query_id=query.id,
                                           query_mention=query.mention,
                                           start_index=query.startIndex,
                                           end_index=query.endIndex,
                                           answers=list(dpr_answers),
                                           positive_ctxs=[dpr_pos],
                                           negative_ctxs=list(),
                                           hard_negative_ctxs=dpr_neg_ctxs))
        # else:
        #     dpr_examples.append(DPRExample(dataset="WEC",
        #                                    question=dpr_question,
        #                                    query_id=query.id,
        #                                    query_mention=query.mention,
        #                                    start_index=query.startIndex,
        #                                    end_index=query.endIndex,
        #                                    answers=list(dpr_answers),
        #                                    positive_ctxs=dpr_pos_ctxs,
        #                                    negative_ctxs=list(),
        #                                    hard_negative_ctxs=dpr_neg_ctxs))

    with open(dpr_out, 'w', encoding='utf-8') as train_exmpl_os:
        json.dump(dpr_examples, train_exmpl_os, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)

    print("Done!")


if __name__ == '__main__':
    SPLIT = "Train"
    main()
