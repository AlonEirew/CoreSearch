import math

import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.data_obj import EvaluationObject


def evaluate(model, dev_batches, device):
    model.eval()
    evaluation_objects = list()
    for step, batch in enumerate(dev_batches):
        batch = tuple(t.to(device) for t in batch)
        passage_input_ids, query_input_ids, \
        passage_input_mask, query_input_mask, \
        passage_segment_ids, query_segment_ids, \
        passage_event_starts, passage_event_ends, \
        query_event_starts, query_event_ends, \
        passage_end_bound, is_positives = batch

        with torch.no_grad():
            outputs = model(passage_input_ids, query_input_ids,
                            passage_input_mask, query_input_mask,
                            passage_segment_ids, query_segment_ids,
                            passage_event_starts, passage_event_ends)

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

    generate_results_matrics(evaluation_objects)
    # generate_results_inspection(tokenization.tokenizer, evaluation_objects)


def generate_results_matrics(evaluation_objects):
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
    print("Start Position: accuracy={}, precision={}, recall={}, f1={}".format(s_accuracy, s_precision, s_recall, s_f1))
    print("End Position: accuracy={}, precision={}, recall={}, f1={}".format(e_accuracy, e_precision, e_recall, e_f1))
    # print("Avg Position: precision={}, recall={}, f1={}".format(e_precision, e_recall, e_f1))


def generate_results_inspection(tokenizer, evaluation_objects):
    for eval in evaluation_objects:
        selected_start, selected_end = passage_position_selection(eval)
        as_tokens = tokenizer.convert_ids_to_tokens(eval.tokens_ids)
        print("Query event=" + " ".join(as_tokens[eval.query_event_start:eval.query_event_end + 1]))
        print("Gold pass event=" + " ".join(as_tokens[eval.start_label:eval.end_label + 1]))
        print("Pred pass event=" + " ".join(as_tokens[selected_start:selected_end + 1]))


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
