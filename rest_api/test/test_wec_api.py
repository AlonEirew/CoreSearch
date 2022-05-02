import os

import pytest
from haystack.document_stores import BaseDocumentStore
from rest_api.controller.search import query

from rest_api.utils import get_pipelines, get_app
from starlette.testclient import TestClient



@pytest.fixture
def client():
    app = get_app()
    client = TestClient(app)
    pipelines = get_pipelines()
    yield client


@pytest.fixture
def populated_client(client: TestClient):
    pipelines = get_pipelines()
    yield client


def test_query():
    app = get_app()
    client = TestClient(app)
    # query_req = {"query": {"query": "Who is the father of Arya Stark?",
    #                        "query_id": "114977", "start_index": 1, "end_index": 4}, "params": {}}
    query_req = {"query": "Who is the father of Arya Stark?", "params": {}}
    response = client.post(url="/query", json=query_req)
    # result = query(query_req)
    print(response)
