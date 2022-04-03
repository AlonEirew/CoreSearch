import json
import logging
from typing import Dict, List

from haystack.pipelines import DocumentSearchPipeline, ExtractiveQAPipeline
from tqdm import tqdm

from src.data_obj import QueryResult, Passage, BasicMent

logger = logging.getLogger("run_haystack_pipeline")


class WECPipeline(object):
    def __init__(self, document_store, retriever):
        self.document_store = document_store
        self.retriever = retriever
        self.pipeline = self.build_pipeline()

    def build_pipeline(self):
        raise NotImplementedError

    def run_pipeline(self, query_text):
        raise NotImplementedError

    def extract_results(self, query: BasicMent, result: Dict) -> QueryResult:
        raise NotImplementedError

    def run_end_to_end(self, query_examples: List[BasicMent]) -> List[QueryResult]:
        predictions = list()
        for query in tqdm(query_examples, "Running Queries"):
            query_dict = dict()
            query_dict["query"] = " ".join(query.context)
            query_dict["query_mention"] = query.mention
            query_dict["query_id"] = query.id
            query_dict["start_index"] = query.startIndex
            query_dict["end_index"] = query.endIndex
            try:
                # results = self.run_pipeline(json.dumps(query_dict, ensure_ascii=False))
                results = self.run_pipeline(query_text=query_dict)
                query_result = self.extract_results(query, results)
                predictions.append(query_result)
            except TypeError:
                print("Got Error:" + json.dumps(query_examples))
        return predictions


class RetrievalOnlyPipeline(WECPipeline):
    def __init__(self, document_store, retriever, ret_topk):
        super().__init__(document_store, retriever)
        self.ret_topk = ret_topk

    def build_pipeline(self):
        return DocumentSearchPipeline(self.retriever)

    def run_pipeline(self, query_text: str):
        return self.pipeline.run(query=query_text, params={"Retriever": {"top_k": self.ret_topk}})

    def extract_results(self, query: BasicMent, results: Dict) -> QueryResult:
        converted_list = list()
        for result in results["documents"]:
            if query.id != result.id:
                meta = result.meta
                meta["id"] = result.id
                meta["context"] = result.content
                meta["score"] = result.score
                converted_list.append(Passage(meta))

        return QueryResult(query, converted_list)


class QAPipeline(WECPipeline):
    def __init__(self, document_store, retriever, reader, ret_topk, read_topk):
        self.reader = reader
        self.ret_topk = ret_topk
        self.read_topk = read_topk
        super().__init__(document_store, retriever)

    def build_pipeline(self):
        return ExtractiveQAPipeline(self.reader, self.retriever)

    def run_pipeline(self, query_text):
        return self.pipeline.run(
            query=query_text, params={"Retriever": {"top_k": self.ret_topk}, "Reader": {"top_k": self.read_topk}}
        )

    def extract_results(self, query: BasicMent, results: Dict) -> QueryResult:
        converted_list = list()
        for result in results["answers"]:
            ans_id = result.document_id
            if query.id != ans_id:
                ans_doc = next((doc for doc in results["documents"] if doc.id == ans_id), None)
                assert ans_doc
                meta = ans_doc.meta
                meta["id"] = ans_id
                meta["context"] = ans_doc.content
                meta["score"] = ans_doc.score
                meta["answer"] = result.answer
                meta["offsets_in_document"] = result.offsets_in_document
                converted_list.append(Passage(meta))
        return QueryResult(query, converted_list)
