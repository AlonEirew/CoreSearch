import json
from os import path

from src.override_classes.wec_bm25_processor import WECBM25Processor
from src.override_classes.wec_context_processor import WECContextProcessor
from src.utils import dpr_utils
from src.utils.io_utils import write_json


def train():
    parameters = dict()
    parameters["doc_dir"] = "data/resources/dpr/context_full_queries_permut/"
    parameters["train_filename"] = "Train_ctx_format.json"
    parameters["dev_filename"] = "Dev_ctx_format_false.json"
    parameters["model_str"] = "dev_spanbert_hidden_cls_full_ctx_2it"
    parameters["query_style"] = "context"
    parameters["add_spatial_tokens"] = False

    parameters["n_epochs"] = 2
    parameters["max_seq_len_query"] = 64
    parameters["max_seq_len_passage"] = 180
    parameters["batch_size"] = 16

    parameters["query_model"] = "SpanBERT/spanbert-base-cased"
    parameters["passage_model"] = "SpanBERT/spanbert-base-cased"
    parameters["faiss_path_prefix"] = "indexes/spanbert_notft/dev_index"

    parameters["out_model_name"] = "dev_" + parameters["model_str"] + "_" + str(parameters["n_epochs"]) + "it"

    parameters["note"] = "Experiment running with DPR file containing all the queries " \
                         "(taking each query with all its positive passage), using cls from last hidden layer"
    checkpoint_dir = "data/checkpoints/" + parameters["out_model_name"]

    print("Run Values:\n" + str(json.dumps(parameters, default=lambda o: o.__dict__, indent=4, sort_keys=True)))
    evaluate_every = 350
    run(parameters, checkpoint_dir, evaluate_every)


def run(parameters, checkpoint_dir, evaluate_every):
    query_style = parameters["query_style"]
    if query_style == "bm25":
        processor_type = WECBM25Processor
    elif query_style == "context":
        processor_type = WECContextProcessor
    else:
        raise TypeError(f"No processor that support {query_style}")

    faiss_index_path = "%s.faiss" % parameters["faiss_path_prefix"]
    faiss_config_path = "%s.json" % parameters["faiss_path_prefix"]
    document_store, retriever = dpr_utils.load_wec_faiss_dpr(faiss_file_path=faiss_index_path,
                                                             faiss_config_file=faiss_config_path,
                                                             query_encode=parameters["query_model"],
                                                             passage_encode=parameters["passage_model"],
                                                             infer_tokenizer_classes=True,
                                                             max_seq_len_query=parameters["max_seq_len_query"],
                                                             max_seq_len_passage=parameters["max_seq_len_passage"],
                                                             batch_size=parameters["batch_size"],
                                                             processor_type=processor_type,
                                                             add_spatial_tokens=parameters["add_spatial_tokens"])

    retriever.train(data_dir=parameters["doc_dir"],
                    train_filename=parameters["train_filename"],
                    dev_filename=parameters["dev_filename"],
                    test_filename=parameters["dev_filename"],
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
