import json
from os import path

from src.utils import dpr_utils
from src.utils.io_utils import write_json


def train():
    parameters = dict()
    parameters["doc_dir"] = "data/resources/dpr/ment_ners/"
    parameters["train_filename"] = "Train_dpr_format.json"
    parameters["dev_filename"] = "Dev_dpr_format.json"
    parameters["model_str"] = "spanbert_bm25"

    parameters["n_epochs"] = 2
    parameters["max_seq_len_query"] = 64
    parameters["max_seq_len_passage"] = 180
    parameters["batch_size"] = 16

    parameters["query_model"] = "SpanBERT/spanbert-base-cased"
    parameters["passage_model"] = "SpanBERT/spanbert-base-cased"
    parameters["faiss_path_prefix"] = "indexes/spanbert_notft/dev_index"

    parameters["out_model_name"] = "dev_" + parameters["model_str"] + "_" + str(parameters["n_epochs"]) + "it"

    parameters["note"] = "Experiment running with DPR file containing onlh part of the quires (not taking each query with all its positive passage)"
    checkpoint_dir = "data/checkpoints/" + parameters["out_model_name"]

    evaluate_every = 40
    run(parameters, checkpoint_dir, evaluate_every)


def run(parameters, checkpoint_dir, evaluate_every):

    faiss_index_path = "%s.faiss" % parameters["faiss_path_prefix"]
    faiss_config_path = "%s.json" % parameters["faiss_path_prefix"]
    document_store, retriever = dpr_utils.load_wec_faiss_dpr(faiss_index_path,
                                                             faiss_config_path,
                                                             parameters["query_model"],
                                                             parameters["passage_model"],
                                                             True,
                                                             parameters["max_seq_len_query"],
                                                             parameters["max_seq_len_passage"],
                                                             parameters["batch_size"])

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
