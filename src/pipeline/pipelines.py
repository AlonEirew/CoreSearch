from typing import Dict, List

from haystack.pipeline import ExtractiveQAPipeline, DocumentSearchPipeline
from haystack.utils import print_answers

from src.utils import measurments


class BasePipeline(object):
    def __init__(self, document_store, retriever):
        self.document_store = document_store
        self.retriever = retriever
        self.pipeline = self.build_pipeline()

    def build_pipeline(self):
        raise NotImplementedError

    def run_pipeline(self, query_text):
        raise NotImplementedError

    def extract_results(self, result) -> List[int]:
        raise NotImplementedError

    def run_end_to_end(self, query_examples):
        prediction: Dict[int, List[int]] = dict()
        golds: Dict[int, List[int]] = dict()
        for qid, query in query_examples.items():
            query_text = " ".join(query["context"])
            results = self.run_pipeline(query_text)

            golds[qid] = query["positivePassagesIds"]
            prediction[qid] = list()
            for result in self.extract_results(results):
                if qid != result:
                    prediction[qid].append(result)

            print("query_text=" + query_text)
            # print_answers(results, details="minimal")

        print("MRR@10=" + str(measurments.mean_reciprocal_rank(predictions=prediction, golds=golds, topk=5)))
        print("HIT@10=" + str(measurments.hit_rate(predictions=prediction, golds=golds, topk=5)))


class RetrievalOnlyPipeline(BasePipeline):
    def __init__(self, document_store, retriever, ret_topk):
        super().__init__(document_store, retriever)
        self.ret_topk = ret_topk

    def build_pipeline(self):
        return DocumentSearchPipeline(self.retriever)

    def run_pipeline(self, query_text):
        return self.pipeline.run(
            query=query_text, params={"Retriever": {"top_k": self.ret_topk}}
        )

    def extract_results(self, results):
        converted_list = list()
        for result in results["documents"]:
            converted_list.append(int(result["id"]))
        return converted_list


class QAPipeline(BasePipeline):
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

    def extract_results(self, results):
        converted_list = list()
        for result in results["answers"]:
            converted_list.append(int(result["document_id"]))
        return converted_list
