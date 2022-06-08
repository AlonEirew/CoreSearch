import json
import os
from os import path

os.environ["MILVUS2_ENABLED"] = "false"

from src.override_classes.retriever.wec_dense import WECDensePassageRetriever

from src.override_classes.retriever.wec_bm25_processor import WECBM25Processor
from src.override_classes.retriever.wec_context_processor import WECContextProcessor
from src.utils.io_utils import write_json


def train():
    """
    query_style: bm25 - for bm25 queries, will take the query from beginning till max_query_length
    query_style: context - for context queries, will take the query mention span and surrounding till max_query_length
                           Then this will indicate to the encoder to extract the CLS token
    query_style: start_end - for context queries, will take the query mention span and surrounding till max_query_length
                           Then this will indicate to the encoder to extract the QUERY_START/QUERY_END tokens
    """
    parameters = dict()
    parameters["note"] = "Retriever-Train SpanBERT model, query surrounding context *with* <span> tokens"

    parameters["doc_dir"] = "data/resources/dpr/context/"
    parameters["train_filename"] = "Train.json"
    parameters["dev_filename"] = "Dev.json"

    parameters["model_str"] = "Retriever_SpanBERT"

    parameters["query_style"] = "context"
    parameters["n_epochs"] = 5
    parameters["max_seq_len_query"] = 64
    parameters["max_seq_len_passage"] = 180
    parameters["batch_size"] = 64

    parameters["infer_tokenizer_classes"] = True
    parameters["query_model"] = "SpanBERT/spanbert-base-cased"
    parameters["passage_model"] = "SpanBERT/spanbert-base-cased"

    parameters["out_model_name"] = parameters["model_str"] + "_" + str(parameters["n_epochs"]) + "it"

    checkpoint_dir = "data/checkpoints/" + parameters["out_model_name"]

    print("Run Values:\n" + str(json.dumps(parameters, default=lambda o: o.__dict__, indent=4, sort_keys=True)))
    evaluate_every = 550
    run(parameters, checkpoint_dir, evaluate_every)


def run(parameters, checkpoint_dir, evaluate_every):
    query_style = parameters["query_style"]
    if query_style == "bm25":
        processor_type = WECBM25Processor
        parameters["add_special_tokens"] = False
    elif query_style == "context":
        processor_type = WECContextProcessor
        parameters["add_special_tokens"] = True
    elif query_style == "context_no_toks":
        processor_type = WECContextProcessor
        parameters["add_special_tokens"] = False
    else:
        raise TypeError(f"No processor that support {query_style}")

    retriever = WECDensePassageRetriever(document_store=None, query_embedding_model=parameters["query_model"],
                                         passage_embedding_model=parameters["passage_model"],
                                         infer_tokenizer_classes=parameters["infer_tokenizer_classes"],
                                         max_seq_len_query=parameters["max_seq_len_query"],
                                         max_seq_len_passage=parameters["max_seq_len_passage"],
                                         batch_size=parameters["batch_size"], use_gpu=True, embed_title=False,
                                         use_fast_tokenizers=False, processor_type=processor_type,
                                         add_special_tokens=parameters["add_special_tokens"])

    retriever.train(data_dir=parameters["doc_dir"],
                    train_filename=parameters["train_filename"],
                    dev_filename=parameters["dev_filename"],
                    test_filename=None,
                    n_epochs=parameters["n_epochs"],
                    batch_size=parameters["batch_size"],
                    save_dir=checkpoint_dir,
                    evaluate_every=evaluate_every,
                    embed_title=False,
                    num_positives=1,
                    num_hard_negatives=1,
                    max_processes=5)

    write_json(path.join(checkpoint_dir, "params.json"), parameters)


if __name__ == "__main__":
    train()
