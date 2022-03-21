from typing import List

from src.data_obj import Cluster, TrainExample
from src.pipeline.pipelines import RetrievalOnlyPipeline
from src.pipeline.run_haystack_pipeline import predict_and_eval
from src.utils import io_utils, dpr_utils
from src.utils.io_utils import replace_retriever_model


def train():
    doc_dir = "data/resources/dpr/all_pos_query/"
    train_filename = "Train_bm25_format.json"
    dev_filename = "Dev_bm25_format.json"

    n_epochs = 2
    model_str = "spanbert_bm25"

    max_seq_len_query = 50
    max_seq_len_passage = 150
    batch_size = 16

    query_model = "SpanBERT/spanbert-base-cased"
    passage_model = "SpanBERT/spanbert-base-cased"

    faiss_path_prefix = "indexes/spanbert_notft/dev_index"

    out_model_name = "dev_" + model_str + "_" + str(n_epochs) + "it"
    checkpoint_dir = "data/checkpoints/" + out_model_name

    run(query_model, passage_model, doc_dir, train_filename, dev_filename,
        checkpoint_dir, faiss_path_prefix,
        n_epochs, max_seq_len_query, max_seq_len_passage, batch_size)


def run(query_model, passage_model, doc_dir, train_filename, dev_filename, save_dir, faiss_path_prefix,
        n_epochs, max_seq_len_query, max_seq_len_passage, batch_size):

    faiss_index_path = "%s.faiss" % faiss_path_prefix
    faiss_config_path = "%s.json" % faiss_path_prefix
    document_store, retriever = dpr_utils.load_faiss_dpr(faiss_index_path,
                                                         faiss_config_path,
                                                         query_model,
                                                         passage_model,
                                                         True,
                                                         max_seq_len_query,
                                                         max_seq_len_passage,
                                                         batch_size,
                                                         False)

    replace_retriever_model(retriever,
                            query_model,
                            passage_model,
                            query_model,
                            passage_model,
                            max_seq_len_query,
                            max_seq_len_passage)

    retriever.train(data_dir=doc_dir,
                    train_filename=train_filename,
                    dev_filename=dev_filename,
                    test_filename=dev_filename,
                    n_epochs=n_epochs,
                    batch_size=16,
                    grad_acc_steps=8,
                    save_dir=save_dir,
                    evaluate_every=40,
                    embed_title=False,
                    num_positives=1,
                    num_hard_negatives=1,
                    max_processes=5)


if __name__ == "__main__":
    train()
