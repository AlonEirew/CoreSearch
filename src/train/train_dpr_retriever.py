from typing import List

from src.data_obj import Cluster, TrainExample
from src.pipeline.pipelines import RetrievalOnlyPipeline
from src.pipeline.run_haystack_pipeline import predict_and_eval
from src.utils import io_utils, dpr_utils


def train():
    doc_dir = "data/resources/dpr/bm25_full_queries_permut/"
    train_filename = "Train_bm25_format.json"
    dev_filename = "Dev_bm25_small_format.json"

    run_update_eval = False
    n_epochs = 2
    model_str = "baseline_model"

    infer_tokenizer_classes = False
    max_seq_len_query = 64
    max_seq_len_passage = 180
    batch_size = 16

    # query_model = "bert-base-cased"
    # passage_model = "bert-base-cased"
    # query_model = "SpanBERT/spanbert-base-cased"
    # passage_model = "SpanBERT/spanbert-base-cased"
    query_model = "facebook/dpr-question_encoder-multiset-base"
    passage_model = "facebook/dpr-ctx_encoder-multiset-base"

    faiss_path_prefix = "indexes/multi_notft/dev_index"

    dev_gold_clusters = "data/resources/WEC-ES/Dev_gold_clusters.json"
    dev_train_queries = "data/resources/train/Dev_training_queries.json"

    out_model_name = "dev_" + model_str + "_" + str(n_epochs) + "it"
    checkpoint_dir = "data/checkpoints/" + out_model_name
    result_out_file = "results/" + out_model_name + ".txt"

    run(query_model, passage_model, doc_dir, train_filename, dev_filename,
        checkpoint_dir, faiss_path_prefix, dev_gold_clusters,
        dev_train_queries, n_epochs, result_out_file, run_update_eval, infer_tokenizer_classes,
        max_seq_len_query, max_seq_len_passage, batch_size)


def run(query_model, passage_model, doc_dir, train_filename, dev_filename, save_dir, faiss_path_prefix,
        dev_gold_clusters, dev_train_queries, n_epochs, result_out_file, run_update_eval,
        infer_tokenizer_classes, max_seq_len_query, max_seq_len_passage, batch_size):
    evaluate_every = 350

    faiss_index_path = "%s.faiss" % faiss_path_prefix
    faiss_config_path = "%s.json" % faiss_path_prefix
    document_store, retriever = dpr_utils.load_faiss_dpr(faiss_index_path,
                                                         faiss_config_path,
                                                         query_model,
                                                         passage_model,
                                                         infer_tokenizer_classes,
                                                         max_seq_len_query,
                                                         max_seq_len_passage,
                                                         batch_size)

    retriever.train(data_dir=doc_dir,
                    train_filename=train_filename,
                    dev_filename=dev_filename,
                    test_filename=dev_filename,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    grad_acc_steps=8,
                    save_dir=save_dir,
                    evaluate_every=evaluate_every,
                    embed_title=False,
                    num_positives=1,
                    num_hard_negatives=1,
                    max_processes=5)

    if run_update_eval:
        document_store.update_embeddings(retriever=retriever)
        document_store.save(faiss_index_path)

        pipeline = RetrievalOnlyPipeline(document_store=document_store,
                                         retriever=retriever,
                                         ret_topk=150)

        golds: List[Cluster] = io_utils.read_gold_file(dev_gold_clusters)
        query_examples: List[TrainExample] = io_utils.read_train_example_file(dev_train_queries)
        for query in query_examples:
            query.context = query.bm25_query.split(" ")

        predict_and_eval(pipeline, golds, query_examples, "retriever", result_out_file)


if __name__ == "__main__":
    train()
