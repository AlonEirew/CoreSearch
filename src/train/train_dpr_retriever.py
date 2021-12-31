from typing import List

from src import run_pipeline
from src.data_obj import Cluster, TrainExample
from src.pipeline.pipelines import RetrievalOnlyPipeline
from src.utils import io_utils, dpr_utils


def train():
    doc_dir = "data/resources/dpr/"

    train_filename = "Train_dpr_format.json"
    dev_filename = "Dev_dpr_format.json"

    n_epochs = 1

    query_model = "bert-base-uncased"
    passage_model = "bert-base-uncased"
    # query_model = "SpanBERT/spanbert-base-cased"
    # passage_model = "SpanBERT/spanbert-base-cased"

    faiss_path_prefix = "weces_index_for_test/weces_dev_index"
    faiss_index_path = "%s.faiss" % faiss_path_prefix
    faiss_config_path = "%s.json" % faiss_path_prefix

    save_dir = "data/checkpoints/dpr_bert_" + str(n_epochs) + "it"

    dev_gold_clusters = "data/resources/WEC-ES/Dev_gold_clusters.json"
    dev_train_queries = "data/resources/train/Dev_training_queries.json"

    run(query_model, passage_model, doc_dir, train_filename, dev_filename,
        save_dir, faiss_index_path, faiss_config_path, dev_gold_clusters, dev_train_queries, n_epochs)


def run(query_model, passage_model, doc_dir, train_filename, dev_filename, save_dir, faiss_index_path,
        faiss_config_path, dev_gold_clusters, dev_train_queries, n_epochs):

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
                    evaluate_every=20,
                    embed_title=False,
                    num_positives=1,
                    num_hard_negatives=1
                    )

    document_store.update_embeddings(retriever=retriever)
    document_store.save(faiss_index_path)

    pipeline = RetrievalOnlyPipeline(document_store=document_store,
                                     retriever=retriever,
                                     ret_topk=150)

    golds: List[Cluster] = io_utils.read_gold_file(dev_gold_clusters)
    query_examples: List[TrainExample] = io_utils.read_train_example_file(dev_train_queries)
    for query in query_examples:
        query.context = query.bm25_query.split(" ")

    run_pipeline.predict_and_eval(pipeline, golds, query_examples, "retriever")


if __name__ == "__main__":
    train()
