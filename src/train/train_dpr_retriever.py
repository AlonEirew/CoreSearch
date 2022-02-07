from typing import List

from src.data_obj import Cluster, TrainExample
from src.pipeline.pipelines import RetrievalOnlyPipeline
from src.pipeline.run_haystack_pipeline import predict_and_eval
from src.utils import io_utils, dpr_utils


def train():
    doc_dir = "data/resources/dpr/"

    train_filename = "Train_dpr_format.json"
    dev_filename = "Dev_dpr_format.json"

    n_epochs = 2
    run_update_eval = True
    model_str = "spanbert"
    # query_model = "bert-base-cased"
    # passage_model = "bert-base-cased"
    query_model = "SpanBERT/spanbert-base-cased"
    passage_model = "SpanBERT/spanbert-base-cased"
    # query_model = "facebook/dpr-question_encoder-multiset-base"
    # passage_model = "facebook/dpr-ctx_encoder-multiset-base"

    faiss_path_prefix = "indexes/" + model_str + "_ft/dev_index"
    faiss_index_path = "%s.faiss" % faiss_path_prefix
    faiss_config_path = "%s.json" % faiss_path_prefix

    dev_gold_clusters = "data/resources/WEC-ES/Dev_gold_clusters.json"
    dev_train_queries = "data/resources/train/Dev_training_queries.json"

    out_model_name = "dev_" + model_str + "_" + str(n_epochs) + "it"
    checkpoint_dir = "data/checkpoints/" + out_model_name
    result_out_file = "results/" + out_model_name + ".txt"

    run(query_model, passage_model, doc_dir, train_filename, dev_filename,
        checkpoint_dir, faiss_index_path, faiss_config_path, dev_gold_clusters,
        dev_train_queries, n_epochs, result_out_file, run_update_eval)


def run(query_model, passage_model, doc_dir, train_filename, dev_filename, save_dir, faiss_index_path,
        faiss_config_path, dev_gold_clusters, dev_train_queries, n_epochs, result_out_file, run_update_eval):

    document_store = dpr_utils.load_faiss_doc_store(faiss_index_path, faiss_config_path)
    retriever = dpr_utils.create_default_dpr(document_store, query_model, passage_model)

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
                    multiprocessing_strategy='file_system'
                    )

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
