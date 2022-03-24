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

    # query_style_bm25 if true will replace the query.context with the bm25 query
    permute_all_positive = True
    query_style_bm25 = False
    if query_style_bm25:
        dpr_out = "data/resources/dpr/bm25_full_queries_permut/" + SPLIT + "_bm25_format_" + str(permute_all_positive).lower() + ".json"
    else:
        dpr_out = "data/resources/dpr/context_full_queries_permut/" + SPLIT + "_ctx_format_" + str(permute_all_positive).lower() + ".json"

    assert train_queries
    assert train_passages
    dpr_examples = list()
    for qid, query in tqdm(train_queries.items(), "Converting"):
        positives = query.positive_examples
        negatives = query.negative_examples
        dpr_question = " ".join(query.context)
        if query_style_bm25:
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

        if permute_all_positive:
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
        else:
            dpr_examples.append(DPRExample(dataset="WEC",
                                           question=dpr_question,
                                           query_id=query.id,
                                           query_mention=query.mention,
                                           start_index=query.startIndex,
                                           end_index=query.endIndex,
                                           answers=list(dpr_answers),
                                           positive_ctxs=dpr_pos_ctxs,
                                           negative_ctxs=list(),
                                           hard_negative_ctxs=dpr_neg_ctxs))

    with open(dpr_out, 'w', encoding='utf-8') as train_exmpl_os:
        json.dump(dpr_examples, train_exmpl_os, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)

    print("Done!")


if __name__ == '__main__':
    SPLIT = "Dev"
    main()
