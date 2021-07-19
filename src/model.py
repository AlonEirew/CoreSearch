from typing import Dict

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertForQuestionAnswering, BertTokenizer

from src.data.input_feature import InputFeature


class WecEsModel(nn.Module):
    def __init__(self, hidden_size):
        super(WecEsModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.loss_fct = CrossEntropyLoss()
        self.spanbert = BertForQuestionAnswering.from_pretrained('external/spanbert_hf_base')
        # self.qa_outputs = nn.Linear(hidden_size, 2)
        self.add_tokens()

    def add_tokens(self):
        self.tokenizer.add_tokens(["[QSPAN_START]"])
        self.tokenizer.add_tokens(["[QSPAN_END]"])
        self.spanbert.resize_token_embeddings(len(self.tokenizer))
        # self.spanbert.embeddings.word_embeddings.weight[-1, :] = torch.zeros([self.spanbert.config.hidden_size])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,  start_positions=0, end_positions=0):
        outputs = self.spanbert(input_ids, token_type_ids, attention_mask, output_hidden_states=False,
                                start_positions=start_positions, end_positions=end_positions)
        return outputs.loss

    def convert_example_to_features(self, query_obj: Dict, passage_obj: Dict, max_query_length: int,
                                    max_passage_length: int, is_positive_sample: bool):
        # +3 for the spacial tokens ([CLS] & [SEP])
        max_query_length_exclude = max_query_length - 2
        max_pass_length_exclude = max_passage_length - 1
        max_seq_length = max_query_length + max_passage_length

        query_tokenized = list()
        for word in query_obj["context"]:
            query_tokenized.extend(self.tokenizer.tokenize(word))

        if len(query_tokenized) > max_query_length_exclude:
            query_tokenized = query_tokenized[:max_query_length_exclude]

        query_tokenized.insert(0, "[CLS]")
        query_tokenized.append("[SEP]")

        passage_tokenized = list()
        start_index = passage_obj["startIndex"]
        end_index = passage_obj["endIndex"]
        pass_context = passage_obj["context"]
        start_position = end_position = 0
        for i in range(len(pass_context)):
            word_tokens = self.tokenizer.tokenize(pass_context[i])
            passage_tokenized.extend(word_tokens)
            if is_positive_sample:
                if i == start_index:
                    start_position = len(query_tokenized) + len(passage_tokenized) - len(word_tokens)
                if i == end_index:
                    end_position = len(query_tokenized) + len(passage_tokenized) - len(word_tokens)

        if len(passage_tokenized) > max_pass_length_exclude:
            passage_tokenized = passage_tokenized[:max_pass_length_exclude]

        if end_position > max_pass_length_exclude:
            start_position = 0
            end_position = 0

        passage_tokenized.append("[SEP]")

        input_ids = self.tokenizer.convert_tokens_to_ids(query_tokenized + passage_tokenized)

        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(query_tokenized)
        segment_ids.extend([1] * len(passage_tokenized))

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return InputFeature(input_ids, input_mask, segment_ids, start_position, end_position)
