from typing import List

from haystack.document_store import InMemoryDocumentStore, FAISSDocumentStore
from haystack.retriever import DensePassageRetriever

from src import run_pipeline
from src.data_obj import Cluster, TrainExample
from src.pipeline.pipelines import RetrievalOnlyPipeline
from src.utils import io_utils


def train():
    doc_dir = "resources/dpr/"

    train_filename = "Train_dpr_format.json"
    dev_filename = "Dev_dpr_format.json"

    query_model = "bert-base-uncased"
    passage_model = "bert-base-uncased"

    save_dir = "checkpoints/dpr"
    run(query_model, passage_model, doc_dir, train_filename, dev_filename, save_dir)


def run(query_model, passage_model, doc_dir, train_filename, dev_filename, save_dir):
    faiss_file_path = "wec_dev_index.faiss"
    sql_rul = "sqlite:///weces_dev.db"
    document_store = FAISSDocumentStore.load(faiss_file_path=faiss_file_path,
                                             sql_url=sql_rul,
                                             index="document")

    retriever = DensePassageRetriever(document_store=document_store,
                                      query_embedding_model=query_model,
                                      passage_embedding_model=passage_model,
                                      max_seq_len_query=64,
                                      max_seq_len_passage=180
                                      )

    retriever.train(data_dir=doc_dir,
                    train_filename=train_filename,
                    dev_filename=dev_filename,
                    test_filename=dev_filename,
                    n_epochs=2,
                    batch_size=16,
                    grad_acc_steps=8,
                    save_dir=save_dir,
                    evaluate_every=20,
                    embed_title=False,
                    num_positives=1,
                    num_hard_negatives=1
                    )

    document_store.update_embeddings(retriever=retriever)

    pipeline = RetrievalOnlyPipeline(document_store=document_store,
                                     retriever=retriever,
                                     ret_topk=150)

    golds: List[Cluster] = io_utils.read_gold_file("resources/WEC-ES/Dev_gold_clusters.json")
    query_examples: List[TrainExample] = io_utils.read_train_example_file("resources/train/Dev_training_queries.json")
    for query in query_examples:
        query.context = query.bm25_query.split(" ")

    run_pipeline.predict_and_eval(pipeline, golds, query_examples)


def finetune():
    pass


if __name__ == "__main__":
    train()
    # finetune()
