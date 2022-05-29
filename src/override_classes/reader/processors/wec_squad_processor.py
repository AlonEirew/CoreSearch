import logging
from pathlib import Path
from typing import Optional, Union, List, Dict

import numpy as np
from haystack.modeling.data_handler.processor import SquadProcessor, InferenceProcessor
from haystack.modeling.data_handler.samples import SampleBasket, get_passage_offsets, Sample
from haystack.modeling.model.tokenization import Tokenizer, _get_start_of_word_QA

from src.override_classes.retriever.wec_processor import QUERY_SPAN_END, QUERY_SPAN_START

logger = logging.getLogger(__name__)


class WECSquadProcessor(SquadProcessor):
    def __init__(self, tokenizer, max_seq_len: int, data_dir: Optional[Union[Path, str]], add_special_tokens=False, **kwargs):
        super(WECSquadProcessor, self).__init__(tokenizer, max_seq_len, data_dir, **kwargs)
        self.add_special_tokens = add_special_tokens

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
                           return_baskets:bool = False):
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

        baskets = self.arrange_by_queries(baskets)

        # Convert features into pytorch dataset, this step also removes potential errors during preprocessing
        dataset, tensor_names, baskets = self._create_dataset(baskets)

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
            doc_coref_link = d["title"]
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

                question_text = self.tokenize_query(query_obj)
                tokenized_q = self.tokenizer.encode_plus(question_text, return_offsets_mapping=True,
                                                         return_special_tokens_mask=True, add_special_tokens=False)
                # Extract relevant data
                question_tokenids = tokenized_q["input_ids"]
                question_offsets = [x[0] for x in tokenized_q["offset_mapping"]]
                question_sow = _get_start_of_word_QA(tokenized_q.encodings[0].words)

                query_ment_start = tokenized_q.data['input_ids'].index(self.tokenizer.additional_special_tokens_ids[0])
                query_ment_end = tokenized_q.data['input_ids'].index(self.tokenizer.additional_special_tokens_ids[1])

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

    def tokenize_query(self, query_obj: Dict):
        max_query_len = self.max_query_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        query_ctx = query_obj["query_ctx"]
        query_ment_start = query_obj["query_ment_start"]
        query_ment_end = query_obj["query_ment_end"]
        # query_event_start_ind = query_event_end_ind = 0
        if self.add_special_tokens and QUERY_SPAN_END not in query_ctx and QUERY_SPAN_START not in query_ctx:
            self.add_query_bound(query_ctx, query_ment_start, query_ment_end)

        pointer_start = query_ment_start
        pointer_end = query_ment_end + 2
        query_tokenized_len = 0
        final_query_tokenized = query_ctx[pointer_start:pointer_end+1]
        query_tokenized_len += len(self.tokenizer.encode(" ".join(final_query_tokenized), add_special_tokens=False))
        while query_tokenized_len < max_query_len and (pointer_start > 0 or pointer_end < len(query_ctx) - 1):
            if pointer_end < len(query_ctx) - 1:
                pointer_end += 1
                query_tokenized_len += len(self.tokenizer.encode(query_ctx[pointer_end], add_special_tokens=False))
                # query_tokenized_len += len(self.tokenizer.tokenize(query_ctx[pointer_end]))
                if query_tokenized_len < max_query_len:
                    final_query_tokenized.append(query_ctx[pointer_end])
                else:
                    break

            if pointer_start > 0:
                pointer_start -= 1
                query_tokenized_len += len(self.tokenizer.encode(query_ctx[pointer_start], add_special_tokens=False))
                # query_tokenized_len += len(self.tokenizer.tokenize(query_ctx[pointer_start]))
                if query_tokenized_len < max_query_len:
                    final_query_tokenized.insert(0, query_ctx[pointer_start])
                else:
                    break

        return " ".join(final_query_tokenized)

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
                                    "passage_coref_link": basket.raw["passage_coref_link"],
                                    "query_coref_link": basket.raw["query_coref_link"],
                                    "id": sample_id,
                                    "seq_2_start_t": seq_2_start_t,
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

    def validate_query_toks(self, query_mention, query_id, query_feat, query_tokenized):
        if self.add_special_tokens:
            if QUERY_SPAN_START in self.tokenizer.additional_special_tokens and QUERY_SPAN_END in self.tokenizer.additional_special_tokens:
                if query_feat["input_ids"][query_feat["query_ment_start"]] != self.tokenizer.additional_special_tokens_ids[0] or \
                        query_feat["input_ids"][query_feat["query_ment_end"]] != self.tokenizer.additional_special_tokens_ids[1]:
                    raise AssertionError(f"Query ID={query_id} start/end tokens")
            else:
                raise AssertionError("add_spatial_token=True and no spatial tokens in added_tokens_encoder list!")
            # Assert that mention is equal to the tokenized mention (i.e., mention span is currect)
            if query_mention:
                query_lower = "".join(query_mention)
                token_query_lower = "".join(
                    [s.strip('Ġ') for s in
                     query_tokenized[query_feat["query_ment_start"]:query_feat["query_ment_end"] - 1]])
                if query_lower != token_query_lower:
                    print(f"WARNING:Query ({query_lower}) != tokenized query ({token_query_lower}), ID={query_id}")
        else:
            assert query_feat["input_ids"][
                   query_feat["query_ment_start"]:query_feat["query_ment_end"] + 1] == self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(" ".join(query_mention)))
            # Assert that mention is equal to the tokenized mention (i.e., mention span is currect)
            if query_mention:
                query_lower = "".join(query_mention)
                token_query_lower = "".join([
                    s.strip('Ġ') for s in query_tokenized[query_feat["query_ment_start"]:query_feat["query_ment_end"] + 1]
                ]).lower()

                if query_lower != token_query_lower:
                    print(f"WARNING:Query ({query_lower}) != tokenized query ({token_query_lower})")

    @staticmethod
    def add_query_bound(query_ctx: List[str], start_index: int, end_index: int):
        query_ctx.insert(end_index + 1, QUERY_SPAN_END)
        query_ctx.insert(start_index, QUERY_SPAN_START)

    @staticmethod
    def arrange_by_queries(baskets):
        query_to_passage = dict()
        for sample_bask in baskets:
            query_id = sample_bask.raw['query_id']
            if query_id not in query_to_passage:
                query_to_passage[query_id] = list()

            query_to_passage[query_id].append(sample_bask)

        ret_baskets = list()
        for pass_list in query_to_passage.values():
            ret_baskets.extend(pass_list)

        return ret_baskets
