import json
from typing import List, Dict, Any, Tuple, Optional

import os
import logging
from time import sleep

import requests
import streamlit as st
from tqdm import tqdm

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://143.185.131.229:8081")
STATUS = "initialized"
HS_VERSION = "hs_version"
DOC_REQUEST = "query"
DOC_FEEDBACK = "feedback"
DOC_UPLOAD = "file-upload"

# Retriever models types
DENSE = "Dense (DPR)"
SPARSE = "Sparse (MB25)"


class Query(object):
    def __init__(self, json_obj: Dict = None):
        if json_obj:
            self.id = json_obj["id"]
            self.goldChain = str(json_obj["goldChain"])
            self.mention = json_obj["mention"]
            self.startIndex = json_obj["startIndex"]
            self.endIndex = json_obj["endIndex"]
            self.context = json_obj["context"]


def haystack_is_ready():
    """
    Used to show the "Haystack is loading..." message
    """
    url = f"{API_ENDPOINT}/{STATUS}"
    try:
        if requests.get(url).status_code < 400:
            return True
    except Exception as e:
        logging.exception(e)
        sleep(1)  # To avoid spamming a non-existing endpoint at startup
    return False


@st.cache
def haystack_version():
    """
    Get the Haystack version from the REST API
    """
    url = f"{API_ENDPOINT}/{HS_VERSION}"
    return requests.get(url, timeout=0.1).json()


def query(query, filters={}, top_k_reader=5, top_k_retriever=5, retriever_model=DENSE) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    query_dict = dict()
    query_dict["query"] = " ".join(query.context)
    query_dict["query_mention"] = query.mention
    query_dict["query_id"] = query.id
    query_dict["start_index"] = query.startIndex
    query_dict["end_index"] = query.endIndex
    query_dict["query_coref_link"] = -1
    query_dict["model"] = retriever_model

    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    print(query_dict)
    params = {"Retriever": {"top_k": top_k_retriever}, "Reader": {"top_k": top_k_reader}}
    req = {"query": query_dict, "params": params}
    print(req)

    response_raw = requests.post(url, json=req)

    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        raise Exception(", ".join(response["errors"]))

    # Format response
    results = []
    answers = response["answers"]
    for answer in answers:
        ans_document = [doc for doc in response["documents"] if doc["id"] == answer["document_id"]][0]
        if answer.get("answer", None):
            results.append(
                {
                    "context": ans_document["content"],
                    "answer": answer.get("answer", None),
                    "source": answer["meta"]["title"],
                    "relevance": round(answer["score"] * 100, 2),
                    "document": [doc for doc in response["documents"] if doc["id"] == answer["document_id"]][0],
                    "offset_start_in_doc": answer["offsets_in_document"][0]["start"],
                    "_raw": answer,
                }
            )
        else:
            results.append(
                {
                    "context": None,
                    "answer": None,
                    "document": None,
                    "relevance": round(answer["score"] * 100, 2),
                    "_raw": answer,
                }
            )
    return results, response


def send_feedback(query, answer_obj, is_correct_answer, is_correct_document, document) -> None:
    """
    Send a feedback (label) to the REST API
    """
    url = f"{API_ENDPOINT}/{DOC_FEEDBACK}"
    req = {
        "query": query,
        "document": document,
        "is_correct_answer": is_correct_answer,
        "is_correct_document": is_correct_document,
        "origin": "user-feedback",
        "answer": answer_obj,
    }
    response_raw = requests.post(url, json=req)
    if response_raw.status_code >= 400:
        raise ValueError(f"An error was returned [code {response_raw.status_code}]: {response_raw.json()}")


def upload_doc(file):
    url = f"{API_ENDPOINT}/{DOC_UPLOAD}"
    files = [("files", file)]
    response = requests.post(url, files=files).json()
    return response


def get_backlink(result) -> Tuple[Optional[str], Optional[str]]:
    if result.get("document", None):
        doc = result["document"]
        if isinstance(doc, dict):
            if doc.get("meta", None):
                if isinstance(doc["meta"], dict):
                    if doc["meta"].get("url", None) and doc["meta"].get("title", None):
                        return doc["meta"]["url"], doc["meta"]["title"]
    return None, None


def load_json_file(json_file: str):
    assert json_file
    with open(json_file, "r") as fis:
        return json.load(fis)


def read_query_file(queries_file: str) -> List[Query]:
    queries_json = load_json_file(queries_file)
    queries = list()
    for query_obj in tqdm(queries_json, desc="Reading queries"):
        queries.append(Query(query_obj))
    return queries
