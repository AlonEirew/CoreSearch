from typing import Dict, Any

import logging
import time
import json
from numpy import ndarray

from pydantic import BaseConfig
from fastapi import FastAPI, APIRouter
import haystack
from haystack import Pipeline

from rest_api.utils import get_app, get_pipelines, DENSE, SPARSE
from rest_api.schema import QueryRequest, QueryResponse


#logging.getLogger("haystack")
# logging.getLogger(__name__).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BaseConfig.arbitrary_types_allowed = True


router = APIRouter()
app: FastAPI = get_app()
dense_pipeline: Pipeline = get_pipelines().get("dense_pipeline", None)
sparse_pipeline: Pipeline = get_pipelines().get("sparse_pipeline", None)
concurrency_limiter = get_pipelines().get("concurrency_limiter", None)


@router.get("/initialized")
def check_status():
    """
    This endpoint can be used during startup to understand if the
    server is ready to take any requests, or is still loading.

    The recommended approach is to call this endpoint with a short timeout,
    like 500ms, and in case of no reply, consider the server busy.
    """
    return True


@router.get("/hs_version")
def haystack_version():
    """
    Get the running Haystack version.
    """
    docs_count = sparse_pipeline.get_document_store().get_document_count()
    return {"hs_version": haystack.__version__, "total_docs": docs_count}


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(request: QueryRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """
    logger.info(f"Got query to process-{request.query}")
    req_query = request.query
    ret_model = req_query["model"]
    if ret_model == DENSE:
        pipeline = dense_pipeline
    elif ret_model == SPARSE:
        pipeline = sparse_pipeline
    else:
        raise ValueError(f"No supported retriever model for {ret_model}")

    with concurrency_limiter.run():
        result = _process_request(pipeline, request)
        return result


# @send_event_if_public_demo
def _process_request(pipeline, request) -> Dict[str, Any]:
    start_time = time.time()

    params = request.params or {}

    # format global, top-level filters (e.g. "params": {"filters": {"name": ["some"]}})
    if "filters" in params.keys():
        params["filters"] = _format_filters(params["filters"])

    # format targeted node filters (e.g. "params": {"Retriever": {"filters": {"value"}}})
    for key in params.keys():
        if "filters" in params[key].keys():
            params[key]["filters"] = _format_filters(params[key]["filters"])

    result = pipeline.run(query=request.query, params=params, debug=request.debug)

    # Ensure answers and documents exist, even if they're empty lists
    if not "documents" in result:
        result["documents"] = []
    if not "answers" in result:
        result["answers"] = []

    # if any of the documents contains an embedding as an ndarray the latter needs to be converted to list of float
    for document in result["documents"]:
        if isinstance(document.embedding, ndarray):
            document.embedding = document.embedding.tolist()

    logger.info(
        json.dumps({"request": request, "response": result, "time": f"{(time.time() - start_time):.2f}"}, default=str)
    )
    return result


def _format_filters(filters):
    """
    Adjust filters to compliant format:
    Put filter values into a list and remove filters with null value.
    """
    new_filters = {}
    if filters is None:
        logger.warning(
            f"Request with deprecated filter format ('\"filters\": null'). "
            f"Remove empty filters from params to be compliant with future versions"
        )
    else:
        for key, values in filters.items():
            if values is None:
                logger.warning(
                    f"Request with deprecated filter format ('{key}: null'). "
                    f"Remove null values from filters to be compliant with future versions"
                )
                continue

            if not isinstance(values, list):
                logger.warning(
                    f"Request with deprecated filter format ('{key}': {values}). "
                    f"Change to '{key}':[{values}]' to be compliant with future versions"
                )
                values = [values]

            new_filters[key] = values
    return new_filters
