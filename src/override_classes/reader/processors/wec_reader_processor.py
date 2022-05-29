from pathlib import Path
from typing import Optional, Union, List
import logging

from haystack.modeling.data_handler.samples import Sample, SampleBasket, get_passage_offsets

from src.override_classes.reader.processors.wec_squad_processor import WECSquadProcessor

logger = logging.getLogger(__name__)


class WECReaderProcessor(WECSquadProcessor):
    def __init__(self, tokenizer, max_seq_len: int, data_dir: Optional[Union[Path, str]], add_special_tokens=False, **kwargs):
        super(WECReaderProcessor, self).__init__(tokenizer, max_seq_len, data_dir, **kwargs)
        self.add_special_tokens = add_special_tokens

    def _split_docs_into_passages(self, baskets: List[SampleBasket]):
        """
        Because of the sequence length limitation of Language Models, the documents need to be divided into smaller
        parts that we call passages.
        """
        for basket in baskets:
            samples = []
            ########## perform some basic checking
            # TODO, eventually move checking into input validation functions
            # ignore samples with empty context
            if basket.raw["document_text"] == "":
                logger.warning("Ignoring sample with empty context")
                continue
            ########## end checking

            # passage_spans is a list of dictionaries where each defines the start and end of each passage
            # on both token and character level
            try:
                passage_spans = get_passage_offsets(basket.raw["document_offsets"],
                                                    self.max_seq_len-self.max_query_length,
                                                    self.max_seq_len-self.max_query_length,
                                                    basket.raw["document_text"])
            except Exception as e:
                logger.warning(f"Could not devide document into passages. Document: {basket.raw['document_text'][:200]}\n"
                               f"With error: {e}")
                passage_spans = []

            assert len(passage_spans) == 1
            for passage_span in passage_spans:
                # Unpack each variable in the dictionary. The "_t" and "_c" indicate
                # whether the index is on the token or character level
                passage_start_t = passage_span["passage_start_t"]
                passage_end_t = passage_span["passage_end_t"]
                passage_start_c = passage_span["passage_start_c"]
                passage_end_c = passage_span["passage_end_c"]

                passage_start_of_word = basket.raw["document_start_of_word"][passage_start_t: passage_end_t]
                passage_tokens = basket.raw["document_tokens"][passage_start_t: passage_end_t]
                passage_text = basket.raw["document_text"][passage_start_c: passage_end_c]

                clear_text = {"passage_text": passage_text,
                              "question_text": basket.raw["question_text"],
                              "passage_id": passage_span["passage_id"],
                              }
                tokenized = {"passage_start_t": passage_start_t,
                             "passage_start_c": passage_start_c,
                             "passage_tokens": passage_tokens,
                             "passage_start_of_word": passage_start_of_word,
                             "question_tokens": basket.raw["question_tokens"][:self.max_query_length],
                             "question_offsets": basket.raw["question_offsets"][:self.max_query_length],
                             "question_start_of_word": basket.raw["question_start_of_word"][:self.max_query_length],
                             }
                # The sample ID consists of internal_id and a passage numbering
                sample_id = f"{basket.id_internal}-{passage_span['passage_id']}"
                samples.append(Sample(id=sample_id,
                                      clear_text=clear_text,
                                      tokenized=tokenized))

            assert len(samples) == 1
            basket.samples = samples

        return baskets
