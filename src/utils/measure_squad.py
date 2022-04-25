""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
from typing import List

from src.data_obj import QueryResult, Passage


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def eval_qa(predictions: List[QueryResult]):
    f1 = exact_match = total = 0
    for query_result in predictions:
        qa = query_result.query
        ground_truths = qa.answers
        # top_result = query_result.results[0]
        for top_result in query_result.results:
            if top_result.goldChain == qa.goldChain:
                total += 1
                prediction = top_result.answer
                # Only consider answers that are within the gold span or vice versa
                if is_span_overlap(top_result):
                    is_match = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                    exact_match += is_match
                    f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
                    # span_evaluation(qa, top_result, gold_start_offset, gold_end_offset, is_match)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def is_span_overlap(top_result: Passage):
    gold_start_offset, gold_end_offset = extract_gold_span_idx(top_result)
    # return gold_start_offset == top_result.offsets_in_document[0].start and gold_end_offset == top_result.offsets_in_document[0].end
    return (gold_start_offset <= top_result.offsets_in_document[0].start and
            gold_end_offset >= top_result.offsets_in_document[0].end) or \
           (gold_start_offset >= top_result.offsets_in_document[0].start and
            gold_end_offset <= top_result.offsets_in_document[0].end)


def extract_gold_span_idx(top_result):
    context_splt = top_result.context.split(" ")
    context_splt_start_str = context_splt[0:int(top_result.startIndex)]
    context_splt_end_str = context_splt[int(top_result.startIndex):int(top_result.endIndex) + 1]
    gold_start_offset = sum([len(tok) for tok in context_splt_start_str]) + len(context_splt_start_str)
    delta = sum([len(tok) for tok in context_splt_end_str]) + len(context_splt_end_str)
    gold_end_offset = gold_start_offset + delta - 1
    return gold_start_offset, gold_end_offset


def span_evaluation(query, top_result, gold_start_offset, gold_end_offset, exact_match):
    if len(top_result.offsets_in_document) > 1:
        print("More then one offset")
    else:
        if gold_start_offset == top_result.offsets_in_document[0].start and gold_end_offset == \
                top_result.offsets_in_document[0].end:
            return

        print("---------------------------------------")
        print("\tQUERY_ID=" + query.id)
        print("\tQUERY_QUESTION=" + query.bm25_query)
        print("\tQUERY_ANSWERS=" + str(query.answers))
        print()
        print("\tRESULT_ID=" + top_result.id + ":")
        print("\tRESULT_CONTEXT=" + top_result.context)
        print("\tRESULT_ANSWER=" + top_result.answer)
        print("\tRESULT_ANS_OFFSET=(" + str(top_result.offsets_in_document[0].start) + "," + str(
            top_result.offsets_in_document[0].end) + ")")
        print("\tRESULT_GOLD_MENTION=" + top_result.mention)
        print("\tRESULT_GOLD_OFFSET=(" + str(gold_start_offset) + "," + str(gold_end_offset) + ")")
        print()
        print("\tEXACT_MATCH=" + str(exact_match))
        print("---------------------------------------")


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}
