from typing import List, Dict


def hit_rate(predictions: Dict[str, List[str]], golds: Dict[str, List[str]], topk: int):
    assert len(predictions) == len(golds)
    hit_rates = list()
    for query_id in predictions.keys():
        predict_hits = list()
        max_topk = topk if topk <= len(predictions[query_id]) else len(predictions[query_id])
        query_predictions = predictions[query_id][0:max_topk]
        for pass_id in golds[query_id]:
            if pass_id in query_predictions:
                predict_hits.append(pass_id)
        hit_rates.append(len(predict_hits) / len(golds[query_id]))

    return sum(hit_rates) / len(golds)


def mean_reciprocal_rank(predictions: Dict[str, List[str]], golds: Dict[str, List[str]], topk: int):
    assert len(predictions) == len(golds)
    mrr_topk = list()
    for query_id in predictions.keys():
        max_topk = topk if topk <= len(predictions[query_id]) else len(predictions[query_id])
        for index in range(max_topk):
            if predictions[query_id][index] in golds[query_id]:
                mrr_topk.append(1 / (index + 1))
                break

    return sum(mrr_topk) / len(predictions)
