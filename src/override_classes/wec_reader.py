from typing import List, Optional

from haystack.nodes import FARMReader, BaseReader
from haystack.schema import Document, MultiLabel


class WECReader(FARMReader):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)

    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None,
            labels: Optional[MultiLabel] = None, add_isolated_node_eval: bool = False):  # type: ignore
        if isinstance(query, dict):
            query = query["query"]

        self.query_count += 1
        if documents:
            predict = self.timing(self.predict, "query_time")
            results = predict(query=query, documents=documents, top_k=top_k)
        else:
            results = {"answers": []}

        # Add corresponding document_name and more meta data, if an answer contains the document_id
        results["answers"] = [BaseReader.add_doc_meta_data_to_answer(documents=documents, answer=answer) for answer in results["answers"]]

        # run evaluation with labels as node inputs
        if add_isolated_node_eval and labels is not None:
            relevant_documents = [label.document for label in labels.labels]
            results_label_input = predict(query=query, documents=relevant_documents, top_k=top_k)

            # Add corresponding document_name and more meta data, if an answer contains the document_id
            results["answers_isolated"] = [BaseReader.add_doc_meta_data_to_answer(documents=documents, answer=answer) for answer in results_label_input["answers"]]

        return results, "output_1"
