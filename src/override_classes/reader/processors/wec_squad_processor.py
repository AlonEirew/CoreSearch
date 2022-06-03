import logging
import random
from pathlib import Path
from typing import Optional, Union, List, Dict

import numpy as np

from haystack.modeling.data_handler.dataset import convert_features_to_dataset
from haystack.modeling.data_handler.processor import SquadProcessor, InferenceProcessor
from haystack.modeling.data_handler.samples import SampleBasket, offset_to_token_idx_vecorized, Sample
from haystack.modeling.model.tokenization import Tokenizer, _get_start_of_word_QA
from src.override_classes.retriever.wec_processor import QUERY_SPAN_END, QUERY_SPAN_START

logger = logging.getLogger(__name__)


class WECSquadProcessor(SquadProcessor):
    def __init__(self, tokenizer, max_seq_len: int, data_dir: Optional[Union[Path, str]], add_special_tokens=False,
                 num_positives: int = 1, num_negatives: int = 23, batch_size: int = 12, **kwargs):
        super(WECSquadProcessor, self).__init__(tokenizer, max_seq_len, data_dir, **kwargs)
        self.add_special_tokens = add_special_tokens
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.batch_size = batch_size

    @classmethod
    def convert_from_transformers(cls, tokenizer_name_or_path, task_type, max_seq_len, doc_stride,
                                  revision=None, tokenizer_class=None, tokenizer_args=None, use_fast=True,
                                  add_special_tokens=False, **kwargs):
        tokenizer_args = tokenizer_args or {}
        tokenizer = Tokenizer.load(tokenizer_name_or_path,
                                   tokenizer_class=tokenizer_class,
                                   use_fast=use_fast,
                                   revision=revision,
                                   **tokenizer_args,
                                   **kwargs
                                   )

        # TODO infer task_type automatically from config (if possible)
        if task_type == "question_answering":
            processor = WECSquadProcessor(
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                label_list=["start_token", "end_token"],
                metric="squad",
                data_dir="data",
                doc_stride=doc_stride,
                add_special_tokens=add_special_tokens
            )
        elif task_type == "embeddings":
            processor = InferenceProcessor(tokenizer=tokenizer, max_seq_len=max_seq_len)
        else:
            raise ValueError(f"`task_type` {task_type} is not supported yet. "
                             f"Valid options for arg `task_type`: 'question_answering', "
                             f"'embeddings', ")

        return processor

    def dataset_from_dicts(self,
                           dicts: List[dict],
                           indices: Optional[List[int]] = None,
                           return_baskets: bool = False):
        """
        Convert input dictionaries into a pytorch dataset for Question Answering.
        For this we have an internal representation called "baskets".
        Each basket is a question-document pair.
        Each stage adds or transforms specific information to our baskets.

        :param dicts: dict, input dictionary with SQuAD style information present
        :param indices: list, indices used during multiprocessing so that IDs assigned to our baskets is unique
        :param return_baskets: boolean, whether to return the baskets or not (baskets are needed during inference)
        """
        # Convert to standard format
        pre_baskets = [self.convert_qa_input_dict(x) for x in dicts] # TODO move to input object conversion

        # Tokenize documents and questions
        baskets = self.tokenize_batch_question_answering(pre_baskets, indices)

        # Split documents into smaller passages to fit max_seq_len
        baskets = self._split_docs_into_passages(baskets)

        # Convert answers from string to token space, skip this step for inference
        if not return_baskets:
            baskets = self._convert_answers(baskets)

        # Convert internal representation (nested baskets + samples with mixed types) to pytorch features (arrays of numbers)
        baskets = self._passages_to_pytorch_features(baskets, return_baskets)

        if not return_baskets:
            return baskets, return_baskets, indices

        return self.convert_features_to_dataset(baskets, return_baskets, indices)

    def convert_to_final_dataset(self, baskets: List[SampleBasket], return_baskets: bool = False,
                                 indices: Optional[List[int]] = None):
        if not return_baskets:
            baskets = self.arrange_by_queries(baskets)

        # Convert features into pytorch dataset, this step also removes potential errors during preprocessing
        dataset, tensor_names, baskets = self.create_dataset(baskets, return_baskets)

        # Logging
        if indices:
            if 0 in indices:
                self._log_samples(n_samples=1, baskets=self.baskets)

        # During inference we need to keep the information contained in baskets.
        if return_baskets:
            return dataset, tensor_names, self.problematic_sample_ids, baskets
        else:
            return dataset, tensor_names, self.problematic_sample_ids

    def tokenize_batch_question_answering(self, pre_baskets, indices):
        assert len(indices) == len(pre_baskets)
        assert self.tokenizer.is_fast, "Processing QA data is only supported with fast tokenizers for now.\n" \
                                  "Please load Tokenizers with 'use_fast=True' option."
        baskets = []
        # # Tokenize texts in batch mode
        texts = [d["context"] for d in pre_baskets]
        tokenized_docs_batch = self.tokenizer.batch_encode_plus(texts, return_offsets_mapping=True,
                                                                return_special_tokens_mask=True,
                                                                add_special_tokens=False,
                                                                verbose=False)

        # Extract relevant data
        tokenids_batch = tokenized_docs_batch["input_ids"]
        offsets_batch = []
        for o in tokenized_docs_batch["offset_mapping"]:
            offsets_batch.append(np.array([x[0] for x in o]))
        start_of_words_batch = []
        for e in tokenized_docs_batch.encodings:
            start_of_words_batch.append(_get_start_of_word_QA(e.words))

        for i_doc, d in enumerate(pre_baskets):
            doc_coref_link = int(d["title"])
            document_text = d["context"]
            # # Tokenize questions one by one
            for i_q, q in enumerate(d["qas"]):
                if isinstance(q['question'], dict):
                    query_obj = {
                        "query_ctx": q['question']['query'].split(" "),
                        "query_ment_start": q['question']['start_index'],
                        "query_ment_end": q['question']['end_index'],
                        "query_mention": q['question']['query_mention'],
                        "query_id": q['question']['query_id'],
                        "query_coref_link": q['question']['query_coref_link']
                    }
                else:
                    query_obj = {
                        "query_ctx": q['question'].split(" "),
                        "query_ment_start": q['ment_start'],
                        "query_ment_end": q['ment_end'],
                        "query_mention": q['query_mention'],
                        "query_id": q['id'],
                        "query_coref_link": q['query_coref_link']
                    }

                question_text, start_ment_c, ment_len_c = self.tokenize_query(query_obj)
                tokenized_q = self.tokenizer.encode_plus(question_text, return_offsets_mapping=True,
                                                         return_special_tokens_mask=True, add_special_tokens=False)
                # Extract relevant data
                question_tokenids = tokenized_q["input_ids"]
                question_offsets = [x[0] for x in tokenized_q["offset_mapping"]]
                encoded_words = tokenized_q.encodings[0].words
                question_sow = _get_start_of_word_QA(encoded_words)

                if self.add_special_tokens:
                    query_ment_start = tokenized_q.data['input_ids'].index(self.tokenizer.additional_special_tokens_ids[0])
                    query_ment_end = tokenized_q.data['input_ids'].index(self.tokenizer.additional_special_tokens_ids[1])
                else:
                    # Calculate start and end relative to document
                    ment_end_c = start_ment_c + ment_len_c - 1

                    # Convert character offsets to token offsets on document level
                    query_ment_start = offset_to_token_idx_vecorized(np.array(question_offsets), start_ment_c)
                    query_ment_end = offset_to_token_idx_vecorized(np.array(question_offsets), ment_end_c)
                    # in this case the mention will be on the SEP token
                    if query_ment_end == len(tokenized_q.data['input_ids']):
                        query_ment_end -= 1
                    if query_ment_start == len(tokenized_q.data['input_ids']):
                        query_ment_start -= 1

                query_coref_link = query_obj['query_coref_link']

                external_id = q["id"]
                # The internal_id depends on unique ids created for each process before forking
                internal_id = f"{indices[i_doc]}-{i_q}"
                raw = {"document_text": document_text, "document_tokens": tokenids_batch[i_doc],
                       "document_offsets": offsets_batch[i_doc], "document_start_of_word": start_of_words_batch[i_doc],
                       "question_text": question_text, "question_tokens": question_tokenids,
                       "question_offsets": question_offsets, "question_start_of_word": question_sow,
                       "answers": q["answers"], "document_tokens_strings": tokenized_docs_batch.encodings[i_doc].tokens,
                       "question_tokens_strings": tokenized_q.encodings[0].tokens, "query_mention": query_obj['query_mention'],
                       "query_id": query_obj["query_id"], "query_ment_start": query_ment_start,
                       "query_ment_end": query_ment_end, "passage_coref_link": doc_coref_link, "query_coref_link": query_coref_link}
                # TODO add only during debug mode (need to create debug mode)

                baskets.append(SampleBasket(raw=raw, id_internal=internal_id, id_external=external_id, samples=None))
        return baskets

    def _convert_answers(self, baskets:List[SampleBasket]):
        """
        Converts answers that are pure strings into the token based representation with start and end token offset.
        Can handle multiple answers per question document pair as is common for development/text sets
        """
        for basket in baskets:
            error_in_answer = False
            for num, sample in enumerate(basket.samples): # type: ignore
                # Dealing with potentially multiple answers (e.g. Squad dev set)
                # Initializing a numpy array of shape (max_answers, 2), filled with -1 for missing values
                label_idxs = np.full((self.max_answers, 2), fill_value=-1)
                passage_coref_chain = basket.raw['passage_coref_link']

                if error_in_answer or (len(basket.raw["answers"]) == 0):
                    # If there are no answers we set
                    label_idxs[0, :] = 0
                else:
                    # For all other cases we use start and end token indices, that are relative to the passage
                    for i, answer in enumerate(basket.raw["answers"]):
                        # Calculate start and end relative to document
                        answer_len_c = len(answer["text"])
                        answer_start_c = answer["answer_start"]
                        answer_end_c = answer_start_c + answer_len_c - 1

                        # Convert character offsets to token offsets on document level
                        answer_start_t = offset_to_token_idx_vecorized(basket.raw["document_offsets"], answer_start_c)
                        answer_end_t = offset_to_token_idx_vecorized(basket.raw["document_offsets"], answer_end_c)

                        # Adjust token offsets to be relative to the passage
                        answer_start_t -= sample.tokenized["passage_start_t"] # type: ignore
                        answer_end_t -= sample.tokenized["passage_start_t"] # type: ignore

                        # Initialize some basic variables
                        question_len_t = len(sample.tokenized["question_tokens"]) # type: ignore
                        passage_len_t = len(sample.tokenized["passage_tokens"]) # type: ignore

                        # Check that start and end are contained within this passage
                        # answer_end_t is 0 if the first token is the answer
                        # answer_end_t is passage_len_t if the last token is the answer
                        if passage_len_t > answer_start_t >= 0 and passage_len_t >= answer_end_t >= 0:
                            # Then adjust the start and end offsets by adding question and special token
                            label_idxs[i][0] = self.sp_toks_start + question_len_t + self.sp_toks_mid + answer_start_t
                            label_idxs[i][1] = self.sp_toks_start + question_len_t + self.sp_toks_mid + answer_end_t

                        # If the start or end of the span answer is outside the passage, treat passage as no_answer
                        else:
                            label_idxs[i][0] = 0
                            label_idxs[i][1] = 0
                            passage_coref_chain = -1

                        ########## answer checking ##############################
                        # TODO, move this checking into input validation functions and delete wrong examples there
                        # Cases where the answer is not within the current passage will be turned into no answers by the featurization fn
                        if answer_start_t < 0 or answer_end_t >= passage_len_t:
                            pass
                        else:
                            doc_text = basket.raw["document_text"]
                            answer_indices = doc_text[answer_start_c: answer_end_c + 1]
                            answer_text = answer["text"]
                            # check if answer string can be found in context
                            if answer_text not in doc_text:
                                logger.warning(f"Answer '{answer['text']}' not contained in context.\n"
                                               f"Example will not be converted for training/evaluation.")
                                error_in_answer = True
                                label_idxs[i][0] = -100  # TODO remove this hack also from featurization
                                label_idxs[i][1] = -100
                                break  # Break loop around answers, so the error message is not shown multiple times
                            elif answer_indices.strip() != answer_text.strip():
                                logger.warning(f"Answer using start/end indices is '{answer_indices}' while gold label text is '{answer_text}'.\n"
                                               f"Example will not be converted for training/evaluation.")
                                error_in_answer = True
                                label_idxs[i][0] = -100 # TODO remove this hack also from featurization
                                label_idxs[i][1] = -100
                                break # Break loop around answers, so the error message is not shown multiple times
                        ########## end of checking ####################

                sample.tokenized["labels"] = label_idxs # type: ignore
                sample.tokenized["passage_coref_chain"] = passage_coref_chain

        return baskets

    def tokenize_query(self, query_obj: Dict):
        max_query_len = self.max_query_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        query_ctx = query_obj["query_ctx"]
        query_ment_start = query_obj["query_ment_start"]
        query_ment_end = query_obj["query_ment_end"]

        pointer_start = query_ment_start
        pointer_end = query_ment_end
        if self.add_special_tokens and QUERY_SPAN_END not in query_ctx and QUERY_SPAN_START not in query_ctx:
            self.add_query_bound(query_ctx, query_ment_start, query_ment_end)
            pointer_end += 2

        query_tokenized_len = 0
        final_query_tokenized = query_ctx[pointer_start:pointer_end+1]
        query_tokenized_len += len(self.tokenizer.encode(" ".join(final_query_tokenized), add_special_tokens=False))
        start_ment_idx = 0
        ment_len_c = len(" ".join(final_query_tokenized))
        while query_tokenized_len < max_query_len and (pointer_start > 0 or pointer_end < len(query_ctx) - 1):
            if pointer_end < len(query_ctx) - 1:
                pointer_end += 1
                query_tokenized_len += len(self.tokenizer.encode(query_ctx[pointer_end], add_special_tokens=False))
                if query_tokenized_len < max_query_len:
                    final_query_tokenized.append(query_ctx[pointer_end])
                else:
                    break

            if pointer_start > 0:
                pointer_start -= 1
                query_tokenized_len += len(self.tokenizer.encode(query_ctx[pointer_start], add_special_tokens=False))
                if query_tokenized_len < max_query_len:
                    final_query_tokenized.insert(0, query_ctx[pointer_start])
                    start_ment_idx += 1
                else:
                    break

        start_ment_c = len(" ".join(final_query_tokenized[:start_ment_idx])) + 1
        return " ".join(final_query_tokenized), start_ment_c, ment_len_c

    def _passages_to_pytorch_features(self, baskets:List[SampleBasket], return_baskets:bool):
        """
        Convert internal representation (nested baskets + samples with mixed types) to python features (arrays of numbers).
        We first join question and passages into one large vector.
        Then we add vectors for: - input_ids (token ids)
                                 - segment_ids (does a token belong to question or document)
                                 - padding_mask
                                 - span_mask (valid answer tokens)
                                 - start_of_word
        """
        for basket in baskets:
            # Add features to samples
            for num, sample in enumerate(basket.samples): # type: ignore
                # Initialize some basic variables
                if sample.tokenized is not None:
                    question_tokens = sample.tokenized["question_tokens"]
                    question_start_of_word = sample.tokenized["question_start_of_word"]
                    question_len_t = len(question_tokens)
                    passage_start_t = sample.tokenized["passage_start_t"]
                    passage_tokens = sample.tokenized["passage_tokens"]
                    passage_start_of_word = sample.tokenized["passage_start_of_word"]
                    passage_len_t = len(passage_tokens)
                    sample_id = [int(x) for x in sample.id.split("-")]

                    # - Combines question_tokens and passage_tokens into a single vector called input_ids
                    # - input_ids also contains special tokens (e.g. CLS or SEP tokens).
                    # - It will have length = question_len_t + passage_len_t + n_special_tokens. This may be less than
                    #   max_seq_len but never greater since truncation was already performed when the document was chunked into passages
                    question_input_ids = sample.tokenized["question_tokens"]
                    passage_input_ids = sample.tokenized["passage_tokens"]

                input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=question_input_ids,
                                                                            token_ids_1=passage_input_ids)

                segment_ids = self.tokenizer.create_token_type_ids_from_sequences(token_ids_0=question_input_ids,
                                                                                  token_ids_1=passage_input_ids)
                # To make the start index of passage tokens the start manually
                seq_2_start_t = self.sp_toks_start + question_len_t + self.sp_toks_mid

                # at inference this information is missing, need to fill with dummy value
                if "passage_coref_chain" not in sample.tokenized:
                    sample.tokenized["passage_coref_chain"] = -500

                # Verify answer is aligned with label idxs (ugly need to move to function)
                # Only at training, at inference the information is not available
                if len(basket.raw['answers']) > 0:
                    answer_text = basket.raw['answers'][0]["text"]
                    answer_text = "".join(answer_text.split(" ")).lower()
                    if answer_text != '' and sample.tokenized["labels"][0][0] != 0 and sample.tokenized["labels"][0][1] - 1 != 0:
                        start_idx = sample.tokenized["labels"][0][0] - seq_2_start_t
                        end_idx = sample.tokenized["labels"][0][1] + 1 - seq_2_start_t

                        token_answer_lower = "".join(
                            [s.strip('Ġ').lower() for s in
                             self.tokenizer.convert_ids_to_tokens(passage_input_ids[start_idx:end_idx])])
                        if answer_text != token_answer_lower:
                            print(
                                f"WARNING:Answer ({answer_text}) != tokenized answer ({token_answer_lower}), "
                                f"ID={basket.raw['query_id']}")

                start_of_word = [0] * self.sp_toks_start + \
                                question_start_of_word + \
                                [0] * self.sp_toks_mid + \
                                passage_start_of_word + \
                                [0] * self.sp_toks_end

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                padding_mask = [1] * len(input_ids)

                # The span_mask has 1 for tokens that are valid start or end tokens for QA spans.
                # 0s are assigned to question tokens, mid special tokens, end special tokens, and padding
                # Note that start special tokens are assigned 1 since they can be chosen for a no_answer prediction
                span_mask = [1] * self.sp_toks_start
                span_mask += [0] * question_len_t
                span_mask += [0] * self.sp_toks_mid
                span_mask += [1] * passage_len_t
                span_mask += [0] * self.sp_toks_end

                # Pad up to the sequence length. For certain models, the pad token id is not 0 (e.g. Roberta where it is 1)
                pad_idx = self.tokenizer.pad_token_id
                padding = [pad_idx] * (self.max_seq_len - len(input_ids))
                zero_padding = [0] * (self.max_seq_len - len(input_ids))

                input_ids += padding
                padding_mask += zero_padding
                segment_ids += zero_padding
                start_of_word += zero_padding
                span_mask += zero_padding

                # TODO possibly remove these checks after input validation is in place
                len_check = len(input_ids) == len(padding_mask) == len(segment_ids) == len(start_of_word) == len(span_mask)
                id_check = len(sample_id) == 3
                label_check = return_baskets or len(sample.tokenized.get("labels",[])) == self.max_answers # type: ignore
                # labels are set to -100 when answer cannot be found
                label_check2 = return_baskets or np.all(sample.tokenized["labels"] > -99) # type: ignore
                if len_check and id_check and label_check and label_check2:
                    # - The first of the labels will be used in train, and the full array will be used in eval.
                    # - start_of_word and spec_tok_mask are not actually needed by model.forward() but are needed for
                    #   model.formatted_preds() during inference for creating answer strings
                    # - passage_start_t is index of passage's first token relative to document
                    feature_dict = {"input_ids": input_ids,
                                    "padding_mask": padding_mask,
                                    "segment_ids": segment_ids,
                                    "passage_start_t": passage_start_t,
                                    "start_of_word": start_of_word,
                                    "labels": sample.tokenized.get("labels",[]), # type: ignore
                                    "passage_coref_link": sample.tokenized["passage_coref_chain"],
                                    "query_coref_link": basket.raw["query_coref_link"],
                                    "id": sample_id,
                                    "seq_2_start_t": seq_2_start_t,
                                    "span_mask": span_mask,
                                    "query_id": int(basket.raw["query_id"]),
                                    "query_ment_start": basket.raw["query_ment_start"] + 1,
                                    "query_ment_end": basket.raw["query_ment_end"] + 1}

                    self.validate_query_toks(query_mention=basket.raw['query_mention'], query_id=basket.raw['query_id'],
                                             query_feat=feature_dict)
                    # other processor's features can be lists
                    sample.features = [feature_dict] # type: ignore
                else:
                    self.problematic_sample_ids.add(sample.id)
                    sample.features = None
        return baskets

    def create_dataset(self, samples: Union[List[Sample], List[SampleBasket]], return_baskets: bool):
        """
        Convert python features into pytorch dataset.
        Also removes potential errors during preprocessing.
        Flattens nested basket structure to create a flat list of features
        """
        if return_baskets:
            return super()._create_dataset(samples)

        features_flat: List[dict] = []
        if self.check_sample_features(samples):
            for sample in samples: # type: ignore
                features_flat.extend(sample.features) # type: ignore
        else:
            # remove the entire basket
            raise ValueError("Sample is thrown!")

        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names, samples

    @staticmethod
    def check_sample_features(basket: List[Sample]):
        """
        Check if all samples in the basket has computed its features.

        :param basket: the basket containing the samples

        :return: True if all the samples in the basket has computed its features, False otherwise
        """
        if basket is None:
            return False
        elif len(basket) == 0:
            return False
        if basket is None:
            return False
        else:
            for sample in basket:
                if sample.features is None:
                    return False
        return True

    def validate_query_toks(self, query_mention, query_id, query_feat):
        ment_start = query_feat["query_ment_start"]
        ment_end = query_feat["query_ment_end"]
        if self.add_special_tokens:
            if QUERY_SPAN_START in self.tokenizer.additional_special_tokens and QUERY_SPAN_END in self.tokenizer.additional_special_tokens:
                if query_feat["input_ids"][ment_start] != self.tokenizer.additional_special_tokens_ids[0] or \
                        query_feat["input_ids"][ment_end] != self.tokenizer.additional_special_tokens_ids[1]:
                    raise AssertionError(f"Query ID={query_id} start/end tokens")
            else:
                raise AssertionError("add_spatial_token=True and no spatial tokens in added_tokens_encoder list!")
            # Assert that mention is equal to the tokenized mention (i.e., mention span is currect)
            if query_mention:
                query_lower = "".join(query_mention)
                token_query_lower = "".join([
                    s.strip('Ġ') for s in self.tokenizer.convert_ids_to_tokens(
                        query_feat["input_ids"][ment_start:ment_end + 1])
                ]).lower()
                if query_lower != token_query_lower:
                    print(f"WARNING:Query ({query_lower}) != tokenized query ({token_query_lower}), ID={query_id}")
        else:
            # Assert that mention is equal to the tokenized mention (i.e., mention span is currect)
            if query_mention:
                query_lower = "".join(query_mention)
                token_query_lower = "".join([
                    s.strip('Ġ') for s in self.tokenizer.convert_ids_to_tokens(
                        query_feat["input_ids"][ment_start:ment_end + 1])
                ])

                if query_lower != token_query_lower:
                    print(f"WARNING:Query ({query_lower}) != tokenized query ({token_query_lower}), ID={query_id}")

    @staticmethod
    def add_query_bound(query_ctx: List[str], start_index: int, end_index: int):
        query_ctx.insert(end_index + 1, QUERY_SPAN_END)
        query_ctx.insert(start_index, QUERY_SPAN_START)

    def arrange_by_queries(self, baskets: List[SampleBasket]):
        query_pos_to_passage = dict()
        query_neg_to_passage = dict()
        set_query_ids = set()
        random.shuffle(baskets)
        for sample_bask in baskets:
            query_id = sample_bask.raw['query_id']
            set_query_ids.add(query_id)
            if query_id not in query_pos_to_passage:
                query_pos_to_passage[query_id] = list()

            if query_id not in query_neg_to_passage:
                query_neg_to_passage[query_id] = list()

            for sample in sample_bask.samples:
                if sample.features[0]['query_coref_link'] == sample.features[0]['passage_coref_link']:
                    query_pos_to_passage[query_id].append(sample)
                else:
                    query_neg_to_passage[query_id].append(sample)

        ret_baskets = list()
        for query_id in set_query_ids:
            query_bask = list()
            positive_len = len(query_pos_to_passage[query_id])
            total_in_batch = self.num_positives + self.num_negatives
            assert positive_len >= self.num_positives
            query_bask.extend(query_pos_to_passage[query_id])
            query_bask.extend(query_neg_to_passage[query_id][:total_in_batch - positive_len])
            assert len(query_bask) % self.batch_size == 0
            ret_baskets.extend(query_bask)

        return ret_baskets
