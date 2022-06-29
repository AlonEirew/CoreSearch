import json
from typing import Optional, Dict, List

from haystack import Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import ElasticsearchRetriever


class CoreSearchElasticsearchRetriever(ElasticsearchRetriever):
    def __init__(self, document_store: ElasticsearchDocumentStore, top_k: int = 10, custom_query: str = None):
        super(CoreSearchElasticsearchRetriever, self).__init__(document_store=document_store, top_k=top_k,
                                                               custom_query=custom_query)
    
    def retrieve(
            self,
            query: str,
            filters: dict = None,
            top_k: Optional[int] = None,
            index: str = None,
            headers: Optional[Dict[str, str]] = None) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = self.document_store.index

        if isinstance(query, str):
            query = json.loads(query)

        documents = self.document_store.query(query["query"], filters, top_k, self.custom_query, index, headers=headers)
        return documents
