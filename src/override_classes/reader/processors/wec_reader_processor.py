import logging
from pathlib import Path
from typing import Optional, Union, List

import numpy as np

from haystack.modeling.data_handler.samples import SampleBasket, offset_to_token_idx_vecorized
from src.override_classes.reader.processors.wec_squad_processor import WECSquadProcessor

logger = logging.getLogger(__name__)


class WECReaderProcessor(WECSquadProcessor):
    def __init__(self, tokenizer, max_seq_len: int, data_dir: Optional[Union[Path, str]], add_special_tokens=False, **kwargs):
        super(WECReaderProcessor, self).__init__(tokenizer, max_seq_len, data_dir, **kwargs)
        self.add_special_tokens = add_special_tokens

    def _passages_to_pytorch_features(self, baskets:List[SampleBasket], return_baskets:bool):
        """
        Convert internal representation (nested baskets + samples with mixed types) to python features (arrays of numbers).
        We first join question and passages into one large vector.
        Then we add vectors for: - input_ids (token ids)
                                 - segment_ids (does a token belong to question or document)
                                 - query_padding_mask
                                 - span_mask (valid answer tokens)
                                 - start_of_word
        """
        for basket in baskets:
            # Add features to samples
            for num, sample in enumerate(basket.samples): # type: ignore
                # Initialize some basic variables
                if sample.tokenized is not None:
                    question_tokens = sample.tokenized["question_tokens"]
                    question_len_t = len(question_tokens)
                    passage_start_t = sample.tokenized["passage_start_t"]
                    passage_tokens = sample.tokenized["passage_tokens"]
                    passage_len_t = len(passage_tokens)
                    sample_id = [int(x) for x in sample.id.split("-")]

                    # - Combines question_tokens and passage_tokens into a single vector called input_ids
                    # - input_ids also contains special tokens (e.g. CLS or SEP tokens).
                    # - It will have length = question_len_t + passage_len_t + n_special_tokens. This may be less than
                    #   max_seq_len but never greater since truncation was already performed when the document was chunked into passages
                    question_input_ids = sample.tokenized["question_tokens"]
                    passage_input_ids = sample.tokenized["passage_tokens"]

                query_input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=question_input_ids)
                passage_input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=passage_input_ids)

                query_segment_ids = self.tokenizer.create_token_type_ids_from_sequences(token_ids_0=question_input_ids)
                passage_segment_ids = self.tokenizer.create_token_type_ids_from_sequences(token_ids_0=passage_input_ids)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                query_padding_mask = [1] * len(query_input_ids)
                passage_padding_mask = [1] * len(passage_input_ids)

                # The span_mask has 1 for tokens that are valid start or end tokens for QA spans.
                # 0s are assigned to question tokens, mid special tokens, end special tokens, and padding
                # Note that start special tokens are assigned 1 since they can be chosen for a no_answer prediction
                span_mask = [1] * self.sp_toks_start
                span_mask += [0] * question_len_t
                span_mask += [0] * self.sp_toks_mid
                span_mask += [1] * passage_len_t
                span_mask += [0] * self.sp_toks_end

                # Pad up to the sequence length. For certain models, the pad token id is not 0 (e.g. Roberta where it is 1)
                max_passage_length = self.max_seq_len - self.max_query_length
                pad_idx = self.tokenizer.pad_token_id
                query_padding = [pad_idx] * (self.max_seq_len - max_passage_length)
                passage_padding = [pad_idx] * (self.max_seq_len - self.max_query_length)
                query_zero_padding = [0] * (self.max_seq_len - max_passage_length)
                passage_zero_padding = [0] * (self.max_seq_len - self.max_query_length)

                query_input_ids += query_padding
                query_padding_mask += query_zero_padding
                query_segment_ids += query_zero_padding

                passage_input_ids += passage_padding
                passage_padding_mask += passage_zero_padding
                passage_segment_ids += passage_zero_padding

                # TODO possibly remove these checks after input validation is in place
                len_check1 = len(query_input_ids) == len(query_padding_mask) == len(query_segment_ids)
                len_check2 = len(passage_input_ids) == len(passage_padding_mask) == len(passage_segment_ids)
                id_check = len(sample_id) == 3
                label_check = return_baskets or len(sample.tokenized.get("labels",[])) == self.max_answers
                # labels are set to -100 when answer cannot be found
                label_check2 = return_baskets or np.all(sample.tokenized["labels"] > -99) # type: ignore
                if len_check1 and id_check and label_check and label_check2:
                    # - The first of the labels will be used in train, and the full array will be used in eval.
                    # - start_of_word and spec_tok_mask are not actually needed by model.forward() but are needed for
                    #   model.formatted_preds() during inference for creating answer strings
                    # - passage_start_t is index of passage's first token relative to document
                    feature_dict = {"input_ids": query_input_ids,
                                    "query_padding_mask": query_padding_mask,
                                    "segment_ids": query_segment_ids,
                                    "passage_start_t": passage_start_t,
                                    "labels": sample.tokenized.get("labels",[]), # type: ignore
                                    "passage_coref_link": basket.raw["passage_coref_link"],
                                    "query_coref_link": basket.raw["query_coref_link"],
                                    "id": sample_id,
                                    "span_mask": span_mask,
                                    "query_ment_start": basket.raw["query_ment_start"] + 1,
                                    "query_ment_end": basket.raw["query_ment_end"] + 1}

                    self.validate_query_toks(query_mention=basket.raw['query_mention'], query_id=basket.raw['query_id'],
                                             query_feat=feature_dict, query_tokenized=basket.raw['question_tokens_strings'])
                    # other processor's features can be lists
                    sample.features = [feature_dict] # type: ignore
                else:
                    self.problematic_sample_ids.add(sample.id)
                    sample.features = None
        return baskets