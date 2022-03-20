import json
from typing import Dict

from tqdm import tqdm

from src.data_obj import TrainExample, Passage, DPRContext, DPRExample
from src.utils import io_utils


def main():
    """
    Generate the DPR examples from all positive quires, meaning for each positive example(i) a query, positive(i)
    will be generated
    """
    train_queries: Dict[str, TrainExample] = TrainExample.list_to_map(io_utils.read_train_example_file("data/resources/train/" + SPLIT + "_training_queries.json"))
    train_passages: Dict[str, Passage] = Passage.list_to_map(io_utils.read_passages_file("data/resources/train/" + SPLIT + "_training_passages.json"))

    dpr_out = "data/resources/dpr/all_pos_query/" + SPLIT + "_bm25_dpr_format.json"

    assert train_queries
    assert train_passages
    dpr_examples = list()
    for qid, query in tqdm(train_queries.items(), "Converting"):
        positives = query.positive_examples
        negatives = query.negative_examples
        dpr_question = query.bm25_query
        dpr_answers = set()
        dpr_pos_ctxs = list()
        dpr_neg_ctxs = list()
        for pos_pass_id in positives:
            pos_passage = train_passages[pos_pass_id]
            dpr_answers.add(" ".join(pos_passage.mention))
            dpr_pos_ctxs.append(DPRContext("NA", " ".join(pos_passage.context), 0, 0, pos_pass_id))
        for neg_pass_id in negatives:
            neg_passage = train_passages[neg_pass_id]
            dpr_neg_ctxs.append(DPRContext("NA", " ".join(neg_passage.context), 0, 0, neg_pass_id))

        for dpr_pos in dpr_pos_ctxs:
            dpr_examples.append(DPRExample("WEC", dpr_question, list(dpr_answers), [dpr_pos], list(), dpr_neg_ctxs))

    with open(dpr_out, 'w', encoding='utf-8') as train_exmpl_os:
        json.dump(dpr_examples, train_exmpl_os, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)

    print("Done!")


if __name__ == '__main__':
    SPLIT = "Dev"
    main()
