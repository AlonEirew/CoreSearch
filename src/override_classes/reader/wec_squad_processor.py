from pathlib import Path
from typing import Optional, Union, List, Dict

import numpy as np
from haystack.modeling.data_handler.processor import SquadProcessor, InferenceProcessor
from haystack.modeling.data_handler.samples import SampleBasket
from haystack.modeling.model.tokenization import Tokenizer, _get_start_of_word_QA

from src.override_classes.retriever.wec_processor import QUERY_SPAN_END, QUERY_SPAN_START


class WECSquadProcessor(SquadProcessor):
    def __init__(self, tokenizer, max_seq_len: int, data_dir: Optional[Union[Path, str]], add_special_tokens=False, **kwargs):
        super().__init__(tokenizer, max_seq_len, data_dir, **kwargs)
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
            document_text = d["context"]
            # # Tokenize questions one by one
            for i_q, q in enumerate(d["qas"]):
                question_text = self.tokenize_query(q)
                tokenized_q = self.tokenizer.encode_plus(question_text, return_offsets_mapping=True,
                                                         return_special_tokens_mask=True, add_special_tokens=False)
                # Extract relevant data
                question_tokenids = tokenized_q["input_ids"]
                question_offsets = [x[0] for x in tokenized_q["offset_mapping"]]
                question_sow = _get_start_of_word_QA(tokenized_q.encodings[0].words)

                external_id = q["id"]
                # The internal_id depends on unique ids created for each process before forking
                internal_id = f"{indices[i_doc]}-{i_q}"
                raw = {"document_text": document_text, "document_tokens": tokenids_batch[i_doc],
                       "document_offsets": offsets_batch[i_doc], "document_start_of_word": start_of_words_batch[i_doc],
                       "question_text": question_text, "question_tokens": question_tokenids,
                       "question_offsets": question_offsets, "question_start_of_word": question_sow,
                       "answers": q["answers"], "document_tokens_strings": tokenized_docs_batch.encodings[i_doc].tokens,
                       "question_tokens_strings": tokenized_q.encodings[0].tokens}
                # TODO add only during debug mode (need to create debug mode)

                baskets.append(SampleBasket(raw=raw, id_internal=internal_id, id_external=external_id, samples=None))
        return baskets

    def tokenize_query(self, query_obj: Dict):
        if isinstance(query_obj['question'], dict):
            query_ctx = query_obj['question']['query'].split(" ")
            ans_ment_start = query_obj['question']['start_index']
            ans_ment_end = query_obj['question']['end_index']
        else:
            query_ctx = query_obj['question'].split(" ")
            ans_ment_start = query_obj['answers'][0]['ment_start']
            ans_ment_end = query_obj['answers'][0]['ment_end']
        query_tokenized = list()
        query_event_start_ind = query_event_end_ind = 0
        if self.add_special_tokens and QUERY_SPAN_END not in query_ctx and QUERY_SPAN_START not in query_ctx:
            self.add_query_bound(query_ctx, ans_ment_start, ans_ment_end)
        for index, word in enumerate(query_ctx):
            query_tokenized.append(word)
            if self.add_special_tokens:
                if word == QUERY_SPAN_START:
                    query_event_start_ind = len(query_tokenized) - 1
                elif word == QUERY_SPAN_END:
                    query_event_end_ind = len(query_tokenized) - 1
            else:
                if index == ans_ment_start:
                    query_event_start_ind = len(query_tokenized) - 1
                if index == ans_ment_end:
                    query_event_end_ind = len(query_tokenized) - 1

        pointer_start = query_event_start_ind
        pointer_end = query_event_end_ind

        if len(query_tokenized) > self.max_query_length:
            trimmed_query_tok = query_tokenized[pointer_start:pointer_end + 1]
            while len(trimmed_query_tok) < self.max_query_length - 1:
                if pointer_end < len(query_tokenized) - 1:
                    pointer_end += 1
                    trimmed_query_tok.append(query_tokenized[pointer_end])
                if pointer_start > 0:
                    pointer_start -= 1
                    trimmed_query_tok.insert(0, query_tokenized[pointer_start])

            query_tokenized = trimmed_query_tok

        return " ".join(query_tokenized)

    @staticmethod
    def add_query_bound(query_ctx: List[str], start_index: int, end_index: int):
        query_ctx.insert(end_index + 1, QUERY_SPAN_END)
        query_ctx.insert(start_index, QUERY_SPAN_START)
