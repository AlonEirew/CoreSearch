import logging
import math

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.data_obj import EvaluationObject

logger = logging.getLogger("event-search")


def generate_span_results(evaluation_objects):
    start_labs, end_labs, start_pred_firsts, end_pred_firsts = (list(), list(), list(), list())
    for eval in evaluation_objects:
        start_labs.append(eval.start_label)
        end_labs.append(eval.end_label)
        start_select, end_select = passage_position_selection(eval)
        # start_pred_firsts.append(eval.start_pred[0][0])
        # end_pred_firsts.append(eval.end_pred[0][0])
        start_pred_firsts.append(start_select)
        end_pred_firsts.append(end_select)
    s_precision, s_recall, s_f1, _ = precision_recall_fscore_support(start_labs, start_pred_firsts, average='macro', zero_division=0)
    e_precision, e_recall, e_f1, _ = precision_recall_fscore_support(end_labs, end_pred_firsts, average='macro', zero_division=0)
    s_accuracy = accuracy_score(start_labs, start_pred_firsts)
    e_accuracy = accuracy_score(end_labs, end_pred_firsts)
    logger.info("Start Position: accuracy={}, precision={}, recall={}, f1={}".format(s_accuracy, s_precision, s_recall, s_f1))
    logger.info("End Position: accuracy={}, precision={}, recall={}, f1={}".format(e_accuracy, e_precision, e_recall, e_f1))
    # print("Avg Position: precision={}, recall={}, f1={}".format(e_precision, e_recall, e_f1))


def generate_results_inspection(tokenizer, evaluation_objects):
    for eval in evaluation_objects:
        selected_start, selected_end = passage_position_selection(eval)
        as_tokens = tokenizer.convert_ids_to_tokens(eval.tokens_ids)
        logger.info("Query event=" + " ".join(as_tokens[eval.query_event_start:eval.query_event_end + 1]))
        logger.info("Gold pass event=" + " ".join(as_tokens[eval.start_label:eval.end_label + 1]))
        logger.info("Pred pass event=" + " ".join(as_tokens[selected_start:selected_end + 1]))


def passage_position_selection(eval_obj):
    # If one of start/end is 0 return 0 for both
    if eval_obj.start_pred[0][0] == 0 or eval_obj.end_pred[0][0] == 0:
        return 0, 0
    # if start position is grater then end position
    if eval_obj.start_pred[0][0] > eval_obj.end_pred[0][0]:
        if eval_obj.start_pred[0][1] > eval_obj.end_pred[0][1]:
            new_end = next((x[0] for x in eval_obj.end_pred if x[0] > eval_obj.start_pred[0][0]), None)
            if new_end:
                return eval_obj.start_pred[0][0], new_end
        elif eval_obj.start_pred[0][1] < eval_obj.end_pred[0][1]:
            new_start = next((x[0] for x in eval_obj.start_pred if x[0] != 0 and x[0] < eval_obj.start_pred[0][0]), None)
            if new_start:
                return new_start, eval_obj.start_pred[0][0]
        return eval_obj.end_pred[0][0], eval_obj.start_pred[0][0]
    return eval_obj.start_pred[0][0], eval_obj.end_pred[0][0]


def generate_sim_results(golds, predictions):
    golds_st = np.concatenate(golds)
    pred_st = np.concatenate(predictions)
    accuracy = accuracy_score(golds_st, pred_st)
    return accuracy


def evaluate_reader(model, dev_batches, similarity_method, negatives, n_gpu):
    model.eval()
    evaluation_objects = list()
    predictions = list()
    gold_labs = list()
    for step, batch in enumerate(dev_batches):
        if n_gpu == 1:
            batch = tuple(t.to(similarity_method.device) for t in batch)
        passage_input_ids, query_input_ids, \
        passage_input_mask, query_input_mask, \
        passage_segment_ids, query_segment_ids, \
        passage_event_starts, passage_event_ends, \
        passage_end_bound, query_event_starts, query_event_ends = batch

        with torch.no_grad():
            outputs = model(passage_input_ids, query_input_ids,
                            passage_input_mask, query_input_mask,
                            passage_segment_ids, query_segment_ids,
                            passage_event_starts, passage_event_ends)
            passage_rep = similarity_method.extract_passage_embeddings(outputs, passage_end_bound)
            query_rep = similarity_method.extract_query_start_end_embeddings(outputs.query_hidden_states, query_event_starts,
                                                                             query_event_ends, negatives + 1)

            passage_rep = passage_rep.view(query_rep.size(0), negatives + 1, -1)
            predicted_idxs, _ = similarity_method.predict_softmax(query_rep, passage_rep)

        predictions.append(predicted_idxs.detach().cpu().numpy())
        golds = np.zeros(len(predicted_idxs))
        # golds[np.arange(0, len(predicted_idxs), negatives+1)] = 1
        gold_labs.append(golds)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        for res_ind in range(start_logits.shape[0]):
            start_tolist = start_logits[res_ind].detach().cpu().numpy()
            end_tolist = end_logits[res_ind].detach().cpu().numpy()
            start_tolist[passage_end_bound[res_ind]:] = -math.inf
            end_tolist[passage_end_bound[res_ind]:] = -math.inf
            top5_start_logits = sorted(enumerate(start_tolist), key=lambda x: x[1], reverse=True)[0:5]
            top5_end_logits = sorted(enumerate(end_tolist), key=lambda x: x[1], reverse=True)[0:5]

            evaluation_objects.append(EvaluationObject(start_label=passage_event_starts.data[res_ind].item(),
                                                       end_label=passage_event_ends.data[res_ind].item(),
                                                       start_pred=top5_start_logits,
                                                       end_pred=top5_end_logits,
                                                       passage_bound=passage_end_bound[res_ind].item(),
                                                       query_event_start=query_event_starts.data[res_ind].item(),
                                                       query_event_end=query_event_ends.data[res_ind].item()))

    generate_span_results(evaluation_objects)
    generate_sim_results(gold_labs, predictions)
    # generate_results_inspection(tokenization.tokenizer, evaluation_objects)


def evaluate_retriever(model, dev_batches, samples, n_gpu):
    model.eval()
    all_predictions = list()
    gold_labs = list()
    for step, batch in enumerate(dev_batches):
        if n_gpu == 1:
            batch = tuple(t.to(model.device) for t in batch)
        passage_input_ids, query_input_ids, \
        passage_input_mask, query_input_mask, \
        passage_segment_ids, query_segment_ids, \
        passage_event_starts, passage_event_ends, \
        passage_end_bound, query_event_starts, query_event_ends = batch

        with torch.no_grad():
            _, predictions = model(passage_input_ids, query_input_ids,
                                   passage_input_mask, query_input_mask,
                                   passage_segment_ids, query_segment_ids,
                                   sample_size=samples)

        all_predictions.append(predictions.detach().cpu().numpy())
        gold_labs.append(np.zeros(len(predictions)))

    accuracy = generate_sim_results(gold_labs, all_predictions)
    logger.info("Dev-Similarity: accuracy={}".format(accuracy))
    return accuracy
