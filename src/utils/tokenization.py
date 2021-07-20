from typing import Dict

from transformers import BertTokenizer

from src.data.input_feature import InputFeature

QUERY_SPAN_START = "[QSPAN_START]"
QUERY_SPAN_END = "[QSPAN_END]"


class Tokenization(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer.add_tokens(QUERY_SPAN_START)
        self.tokenizer.add_tokens(QUERY_SPAN_END)

    def get_tokenizer(self):
        return self.tokenizer

    def convert_example_to_features(self, query_obj: Dict, passage_obj: Dict, max_query_length: int,
                                    max_passage_length: int, is_positive: bool):
        # +3 for the spacial tokens ([CLS] & [SEP])
        max_query_length_exclude = max_query_length - 2
        max_pass_length_exclude = max_passage_length - 1
        max_seq_length = max_query_length + max_passage_length

        passage_start, passage_end, passage_tokenized = self.passage_tokenization(passage_obj, max_pass_length_exclude,
                                                                                  is_positive)
        passage_tokenized.insert(0, "[CLS]")
        if passage_start != 0 and passage_end != 0:
            passage_start += 1
            passage_end += 1

        passage_tokenized.append("[SEP]")
        segment_ids = [0] * len(passage_tokenized)

        query_start, query_end, query_tokenized = self.query_tokenization(query_obj, max_query_length_exclude)
        query_tokenized.append("[SEP]")
        segment_ids.extend([1] * len(query_tokenized))

        all_sequence_tokenized = passage_tokenized + query_tokenized
        if query_start != 0 and query_end != 0:
            query_start += len(passage_tokenized)
            query_end += len(passage_tokenized)

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
                            query_start,
                            query_end,
                            passage_start,
                            passage_end,
                            is_positive)

    def passage_tokenization(self, passage_obj, max_pass_length_exclude, is_positive_sample):
        passage_tokenized = list()
        start_index = passage_obj["startIndex"]
        end_index = passage_obj["endIndex"]
        pass_context = passage_obj["context"]
        passage_start_position = passage_end_position = 0
        for i in range(len(pass_context)):
            word_tokens = self.tokenizer.tokenize(pass_context[i])
            passage_tokenized.extend(word_tokens)
            if is_positive_sample:
                if i == start_index:
                    passage_start_position = len(passage_tokenized) - len(word_tokens)
                if i == end_index:
                    passage_end_position = len(passage_tokenized) - 1

        if len(passage_tokenized) > max_pass_length_exclude:
            passage_tokenized = passage_tokenized[:max_pass_length_exclude]

        if passage_end_position > len(passage_tokenized):
            passage_start_position = 0
            passage_end_position = 0

        assert passage_start_position <= passage_end_position
        return passage_start_position, passage_end_position, passage_tokenized

    def query_tokenization(self, query_obj, max_query_length_exclude):
        query_tokenized = list()
        query_start_position = query_end_position = 0
        for word in query_obj["context"]:
            query_tokenized.extend(self.tokenizer.tokenize(word))
            if word == QUERY_SPAN_START:
                query_start_position = len(query_tokenized)
            elif word == QUERY_SPAN_END:
                query_end_position = len(query_tokenized) - 2

        if len(query_tokenized) > max_query_length_exclude:
            query_tokenized = query_tokenized[:max_query_length_exclude]

        if query_end_position > len(query_tokenized):
            query_start_position = 0
            query_end_position = 0

        return query_start_position, query_end_position, query_tokenized
