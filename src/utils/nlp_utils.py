from typing import Tuple, List

import spacy


class NLPUtils(object):
    spacy_parser = spacy.load('en_core_web_trf')

    def extract_ner_spans(self, context: str) -> List[Tuple[str, str, str, str, str]]:
        doc = self.spacy_parser(context)
        ents = [(e.text, e.lemma_, e.start_char, e.end_char, e.label_) for e in doc.ents]
        return ents
