import logging
from typing import List

from haystack.document_stores import FAISSDocumentStore

from src.data_obj import Cluster, TrainExample, Passage
from src.override_classes.wec_dense import WECDensePassageRetriever
from src.pipeline.pipelines import RetrievalOnlyPipeline
from src.pipeline.run_haystack_pipeline import predict_and_eval
from src.utils import io_utils

logger = logging.getLogger(__name__)


def run(query_model, passage_model, doc_dir, train_filename, dev_filename, save_dir):
    faiss_path_prefix = "weces_index_for_orig_dpr/wec_dev_index"
    faiss_index_file = faiss_path_prefix + ".faiss"
    faiss_config_file = faiss_path_prefix + ".faiss"

    document_store = FAISSDocumentStore.load(index_path=faiss_index_file,
                                             config_path=faiss_config_file)

    retriever = WECDensePassageRetriever(document_store=document_store,
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
    passage_examples: List[Passage] = io_utils.read_passages_file("resources/train/Dev_training_passages.json")
    passage_dict = {obj.id: obj for obj in passage_examples}

    for query in query_examples:
        query.context = query.bm25_query.split(" ")
        for pass_id in query.positive_examples:
            query.answers.add(" ".join(passage_dict[pass_id].mention))

    predict_and_eval(pipeline, golds, query_examples, "retriever")


def train():
    doc_dir = ""

    train_filename = "resources/train/Train_training_queries.json#resources/train/Train_training_passages.json"
    dev_filename = "resources/train/Dev_training_queries.json#resources/train/Dev_training_passages.json"

    query_model = "SpanBERT/spanbert-base-cased"
    passage_model = "SpanBERT/spanbert-base-cased"

    save_dir = "checkpoints/weces_retiever"
    run(query_model, passage_model, doc_dir, train_filename, dev_filename, save_dir)


if __name__ == "__main__":
    train()
