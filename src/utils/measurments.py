from typing import List, Dict


def hit_rate(predictions: Dict[str, List[str]], golds: Dict[str, List[str]], topk: int):
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


def mean_average_precision(predictions: Dict[str, List[str]], golds: Dict[str, List[str]], topk: int):
    app_all = list()
    for query_id in predictions.keys():
        ap_top = list()
        max_topk = topk if topk <= len(predictions[query_id]) else len(predictions[query_id])
        for index in range(max_topk):
            query_pre = precision(predictions[query_id], golds[query_id], index + 1)
            rel = 1 if predictions[query_id][index] in golds[query_id] else 0
            ap_top.append(query_pre * rel)
        app_all.append(sum(ap_top) / len(golds[query_id]))

    return sum(app_all) / len(predictions.keys())


def precision(query_predictions: List[str], query_golds: List[str], till_idx: int):
    true_pos = 0
    for index in range(till_idx):
        if query_predictions[index] in query_golds:
            true_pos += 1

    return true_pos / till_idx


def mean_reciprocal_rank(predictions: Dict[str, List[str]], golds: Dict[str, List[str]], topk: int):
    mrr_topk = list()
    for query_id in predictions.keys():
        max_topk = topk if topk <= len(predictions[query_id]) else len(predictions[query_id])
        for index in range(max_topk):
            if predictions[query_id][index] in golds[query_id]:
                mrr_topk.append(1 / (index + 1))
                break

    return sum(mrr_topk) / len(predictions)


def recall(predictions: Dict[str, List[str]], golds: Dict[str, List[str]], topk: int):
    relevant = 0
    true_pos = 0
    for query_id in predictions.keys():
        max_topk = topk if topk <= len(predictions[query_id]) else len(predictions[query_id])
        relevant += len(golds[query_id])
        for index in range(max_topk):
            if predictions[query_id][index] in golds[query_id]:
                true_pos += 1

    return true_pos / relevant


def accuracy(predictions: Dict[str, List[str]], golds: Dict[str, List[str]], topk: int):
    true_pos = 0
    false_pos = 0
    for query_id in predictions.keys():
        max_topk = topk if topk <= len(predictions[query_id]) else len(predictions[query_id])
        for index in range(max_topk):
            if predictions[query_id][index] in golds[query_id]:
                true_pos += 1
            else:
                false_pos += 1

    return true_pos / (true_pos + false_pos)
