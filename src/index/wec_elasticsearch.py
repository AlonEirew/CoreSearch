from typing import Union, List, Optional, Dict

import numpy as np
from elasticsearch.helpers.actions import bulk
from haystack import Document
from haystack.document_stores import ElasticsearchDocumentStore
from tqdm import tqdm


class WECElasticsearchDocumentStore(ElasticsearchDocumentStore):
    def __init__(
            self,
            host: Union[str, List[str]] = "localhost",
            port: Union[int, List[int]] = 9200,
            username: str = "",
            password: str = "",
            api_key_id: Optional[str] = None,
            api_key: Optional[str] = None,
            aws4auth=None,
            index: str = "document",
            label_index: str = "label",
            search_fields: Union[str, list] = "content",
            content_field: str = "content",
            name_field: str = "name",
            embedding_field: str = "embedding",
            embedding_dim: int = 768,
            custom_mapping: Optional[dict] = None,
            excluded_meta_data: Optional[list] = None,
            analyzer: str = "standard",
            scheme: str = "http",
            ca_certs: Optional[str] = None,
            verify_certs: bool = True,
            create_index: bool = True,
            refresh_type: str = "wait_for",
            similarity="dot_product",
            timeout=30,
            return_embedding: bool = False,
            duplicate_documents: str = 'overwrite',
            index_type: str = "flat",
            scroll: str = "1d",
            skip_missing_embeddings: bool = True,
            synonyms: Optional[List] = None,
            synonym_type: str = "synonym"
    ):
        super(WECElasticsearchDocumentStore, self).__init__(
            host,
            port,
            username,
            password,
            api_key_id,
            api_key,
            aws4auth,
            index,
            label_index,
            search_fields,
            content_field,
            name_field,
            embedding_field,
            embedding_dim,
            custom_mapping,
            excluded_meta_data,
            analyzer,
            scheme,
            ca_certs,
            verify_certs,
            create_index,
            refresh_type,
            similarity,
            timeout,
            return_embedding,
            duplicate_documents,
            index_type,
            scroll,
            skip_missing_embeddings,
            synonyms,
            synonym_type
        )

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,
                        batch_size: int = 10_000, duplicate_documents: Optional[str] = None,
                        headers: Optional[Dict[str, str]] = None):

        if index and not self.client.indices.exists(index=index, headers=headers):
            self._create_document_index(index, headers=headers)

        if index is None:
            index = self.index
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert duplicate_documents in self.duplicate_documents_options, \
            f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(documents=document_objects,
                                                            index=index,
                                                            duplicate_documents=duplicate_documents,
                                                            headers=headers)
        documents_to_index = []
        for doc in tqdm(document_objects, desc="Indexing"):
            _doc = {
                "_op_type": "index" if duplicate_documents == 'overwrite' else "create",
                "_index": index,
                **doc.to_dict(field_map=self._create_document_field_map())
            }  # type: Dict[str, Any]

            # cast embedding type as ES cannot deal with np.array
            if _doc[self.embedding_field] is not None:
                if type(_doc[self.embedding_field]) == np.ndarray:
                    _doc[self.embedding_field] = _doc[self.embedding_field].tolist()

            # rename id for elastic
            _doc["_id"] = str(_doc.pop("id"))

            # don't index query score and empty fields
            _ = _doc.pop("score", None)
            _doc = {k:v for k,v in _doc.items() if v is not None}

            # In order to have a flat structure in elastic + similar behaviour to the other DocumentStores,
            # we "unnest" all value within "meta"
            if "meta" in _doc.keys():
                for k, v in _doc["meta"].items():
                    _doc[k] = v
                _doc.pop("meta")
            documents_to_index.append(_doc)

            # Pass batch_size number of documents to bulk
            if len(documents_to_index) % batch_size == 0:
                bulk(self.client, documents_to_index, request_timeout=300, refresh=self.refresh_type, headers=headers)
                documents_to_index = []

        if documents_to_index:
            bulk(self.client, documents_to_index, request_timeout=300, refresh=self.refresh_type, headers=headers)
