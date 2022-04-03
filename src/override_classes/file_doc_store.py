import json
from abc import ABC
from typing import List, Dict, Any

from haystack import Document
from haystack.document_stores import BaseDocumentStore

from src.data_obj import Passage
from src.utils.io_utils import load_json_file, read_passages_file


class FileDocStore(BaseDocumentStore, ABC):
    def __init__(self, result_file, passages_file, similarity="dot_product"):
        self.results_dict = load_json_file(result_file)
        passage_list: List[Passage] = read_passages_file(passages_file)
        self.passage: Dict[str, Passage] = {obj.id: obj for obj in passage_list}
        self.similarity = similarity

    def get_passages(self, query_id: str, top_k: int) -> List[Document]:
        result_passages: List[Dict] = self.results_dict[query_id]
        top_k = len(result_passages) if top_k > len(result_passages) else top_k
        documents: List[Document] = []
        for result in result_passages[:top_k]:
            passage_id = result["pass_id"]
            score = result["score"]
            raw_passage = self.passage[passage_id]
            meta: Dict[str, Any] = dict()
            meta["mention"] = " ".join(raw_passage.mention)
            meta["startIndex"] = raw_passage.startIndex
            meta["endIndex"] = raw_passage.endIndex
            meta["goldChain"] = raw_passage.goldChain
            documents.append(
                Document(
                    content=" ".join(raw_passage.context),
                    id=passage_id,
                    meta=meta,
                    score=score
                )
            )
        return documents
