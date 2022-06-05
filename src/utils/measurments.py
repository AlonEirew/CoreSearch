from typing import List, Dict

from src.data_obj import QueryResult, Passage
from src.utils.measure_squad import is_span_overlap


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


def mean_average_precision(predictions: List[QueryResult], golds: Dict[str, List[str]], precision_method, topk: int):
    ap_i = list()
    for query_res in predictions:
        query = query_res.query
        idx_rank = list()
        max_topk = topk if topk <= len(query_res.results) else len(query_res.results)
        for index in range(max_topk):
            pred_query_result = query_res.results[index]
            index_prec = precision_method(query_res.results, golds[query.id], index + 1)
            rel = 1 if pred_query_result.id in golds[query.id] else 0
            idx_rank.append(index_prec * rel)
        ap_i.append(sum(idx_rank) / len(golds[query.id]))

    return sum(ap_i) / len(predictions)


def precision(query_predictions: List[Passage], query_golds: List[str], till_idx: int):
    true_pos = 0
    for index in range(till_idx):
        if query_predictions[index].id in query_golds:
            true_pos += 1

    return true_pos / till_idx


def precision_squad(query_predictions: List[Passage], query_golds: List[str], till_idx: int):
    true_pos = 0
    for index in range(till_idx):
        if query_predictions[index].id in query_golds: #and is_span_overlap(query_predictions[index]):
            true_pos += 1

    return true_pos / till_idx


def mean_reciprocal_rank(predictions: List[QueryResult], golds: Dict[str, List[str]], topk: int, method: str):
    mrr_topk = list()
    for query_res in predictions:
        query = query_res.query
        max_topk = topk if topk <= len(query_res.results) else len(query_res.results)
        for index in range(max_topk):
            pred_query_result = query_res.results[index]
            if pred_query_result.id in golds[query.id]:
                # if method == 'retriever':
                mrr_topk.append(1 / (index + 1))
                break
                # elif is_span_overlap(pred_query_result):
                #     mrr_topk.append(1 / (index + 1))
                #     break

    return sum(mrr_topk) / len(predictions)


def recall(predictions: List[QueryResult], golds: Dict[str, List[str]], topk: int):
    relevant = 0
    true_pos = 0
    for query_result in predictions:
        query_id = query_result.query.id
        max_topk = topk if topk <= len(query_result.results) else len(query_result.results)
        relevant += len(golds[query_id])
        for index in range(max_topk):
            if query_result.results[index].id in golds[query_id]:
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
