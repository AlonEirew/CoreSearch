from typing import Dict, List

from tqdm import tqdm
from transformers import BertTokenizer

from src.data_obj import InputFeature
from src.utils import io_utils

QUERY_SPAN_START = "[QSPAN_START]"
QUERY_SPAN_END = "[QSPAN_END]"


class Tokenization(object):
    def __init__(self, model_file=None):
        if model_file:
            self.tokenizer = BertTokenizer.from_pretrained(model_file)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.tokenizer.add_tokens(QUERY_SPAN_START)
            self.tokenizer.add_tokens(QUERY_SPAN_END)

    def read_and_gen_features(self, exmpl_file, passage_file, max_query_length, max_passage_length, remove_qbound) -> List[InputFeature]:
        query_examples = io_utils.read_query_examples_file(exmpl_file)
        print("Done loading examples file-" + exmpl_file)
        passages = io_utils.read_passages_file(passage_file)
        print("Done loading passages file-" + passage_file)

        # list of (query_id, passage_id, pass_ment_bound, gold) where pass_ment_bound = (start_index, end_index)
        positive = list()
        negative = list()
        print("Starting to generate examples...")
        for exmpl in tqdm(query_examples.values()):
            # Generating positive examples
            positive.extend(
                [
                    self.convert_example_to_features(exmpl, passages[pos_id], max_query_length,
                                                             max_passage_length, True, remove_qbound)
                    for pos_id in exmpl["positivePassagesIds"]
                ])

            # Generating negative examples
            negative.extend(
                [
                    self.convert_example_to_features(exmpl, passages[neg_id], max_query_length,
                                                             max_passage_length, False, remove_qbound)
                    for neg_id in exmpl["negativePassagesIds"]
                ])

        print("Done generating data, total of-" + str(len(positive)) + " positive examples")
        print("Done generating data, total of-" + str(len(negative)) + " positive examples")
        return positive + negative

    def convert_example_to_features(self, query_obj: Dict, passage_obj: Dict, max_query_length: int,
                                    max_passage_length: int, is_positive: bool, remove_qbound: bool):
        # +3 for the spacial tokens ([CLS] & [SEP])
        max_query_length_exclude = max_query_length - 2
        max_pass_length_exclude = max_passage_length - 1
        max_seq_length = max_query_length + max_passage_length

        passage_event_start, passage_event_end, passage_tokenized = self.passage_tokenization(passage_obj, max_pass_length_exclude,
                                                                                  is_positive)
        passage_tokenized.insert(0, "[CLS]")
        if passage_event_start != 0 and passage_event_end != 0:
            passage_event_start += 1
            passage_event_end += 1

        passage_tokenized.append("[SEP]")
        segment_ids = [0] * len(passage_tokenized)

        query_event_start, query_event_end, query_tokenized = self.query_tokenization(query_obj, max_query_length_exclude, remove_qbound)
        query_tokenized.append("[SEP]")
        segment_ids.extend([1] * len(query_tokenized))

        all_sequence_tokenized = passage_tokenized + query_tokenized
        if query_event_start != 0 and query_event_end != 0:
            query_event_start += len(passage_tokenized)
            query_event_end += len(passage_tokenized)

        input_ids = self.tokenizer.convert_tokens_to_ids(all_sequence_tokenized)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return InputFeature(input_ids,
                            input_mask,
                            segment_ids,
                            query_obj["id"],
                            passage_obj["id"],
                            query_event_start,
                            query_event_end,
                            len(all_sequence_tokenized),
                            passage_event_start,
                            passage_event_end,
                            len(passage_tokenized),
                            is_positive)

    def passage_tokenization(self, passage_obj, max_pass_length_exclude, is_positive_sample):
        passage_tokenized = list()
        start_index = passage_obj["startIndex"]
        end_index = passage_obj["endIndex"]
        pass_context = passage_obj["context"]
        passage_event_start_ind = passage_event_end_ind = 0
        for i in range(len(pass_context)):
            word_tokens = self.tokenizer.tokenize(pass_context[i])
            passage_tokenized.extend(word_tokens)
            if is_positive_sample:
                if i == start_index:
                    passage_event_start_ind = len(passage_tokenized) - len(word_tokens)
                if i == end_index:
                    passage_event_end_ind = len(passage_tokenized) - 1

        if len(passage_tokenized) > max_pass_length_exclude:
            passage_tokenized = passage_tokenized[:max_pass_length_exclude]

        if passage_event_end_ind > len(passage_tokenized):
            passage_event_start_ind = 0
            passage_event_end_ind = 0

        assert passage_event_start_ind <= passage_event_end_ind
        return passage_event_start_ind, passage_event_end_ind, passage_tokenized

    def query_tokenization(self, query_obj, max_query_length_exclude, remove_qbound):
        query_tokenized = list()
        query_event_start_ind = query_event_end_ind = 0
        for word in query_obj["context"]:
            query_tokenized.extend(self.tokenizer.tokenize(word))
            if word == QUERY_SPAN_START:
                query_event_start_ind = len(query_tokenized)
            elif word == QUERY_SPAN_END:
                query_event_end_ind = len(query_tokenized) - 2

        pointer_start = query_event_start_ind - 1
        pointer_end = query_event_end_ind + 1
        if remove_qbound:
            query_tokenized.remove(QUERY_SPAN_START)
            query_tokenized.remove(QUERY_SPAN_END)
            query_event_start_ind -= 1
            query_event_end_ind -= 1
            pointer_start = query_event_start_ind
            pointer_end = query_event_end_ind

        if len(query_tokenized) > max_query_length_exclude:
            trimmed_query_tok = query_tokenized[pointer_start:pointer_end+1]
            while len(trimmed_query_tok) < max_query_length_exclude - 1:
                if pointer_end < len(query_tokenized) - 1:
                    pointer_end += 1
                    trimmed_query_tok.append(query_tokenized[pointer_end])
                if pointer_start > 0:
                    pointer_start -= 1
                    trimmed_query_tok.insert(0, query_tokenized[pointer_start])

            query_tokenized = trimmed_query_tok
            query_event_start_ind -= pointer_start
            query_event_end_ind -= pointer_start

        assert "".join(query_obj["mention"]) == "".join(
            [s.strip('##') for s in query_tokenized[query_event_start_ind:query_event_end_ind+1]])
        return query_event_start_ind, query_event_end_ind, query_tokenized
