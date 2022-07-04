from pathlib import Path
from typing import List

from haystack import Document
from src.override_classes.retriever.search_context_processor import CoreSearchContextProcessor

from src.utils import io_utils, dpr_utils


def faiss_index(passages_file,
                faiss_file_path,
                sql_rul,
                query_encode,
                passage_encode,
                max_seq_len_query,
                max_seq_len_passage,
                batch_size,
                similarity,
                processor_type,
                add_spatial_tokens,
                faiss_index_factory_str):
    documents: List[Document] = io_utils.read_coresearch_to_haystack_doc_list(passages_file)
    document_store, retriever = dpr_utils.create_coresearch_faiss_dpr(sql_rul,
                                                                      query_encode,
                                                                      passage_encode,
                                                                      True,
                                                                      max_seq_len_query,
                                                                      max_seq_len_passage,
                                                                      batch_size,
                                                                      similarity,
                                                                      processor_type,
                                                                      add_spatial_tokens,
                                                                      faiss_index_factory_str)

    document_store.delete_documents()
    print("Writing document to FAISS index (may take a while)..")
    document_store.write_documents(documents=documents)
    document_store.update_embeddings(retriever=retriever)
    document_store.save(faiss_file_path)
    print("FAISS index creation done!")


def main():
    faiss_dir = "faiss_indexes/best_model"
    faiss_path_prefix = faiss_dir + "/dev_index"
    faiss_file_path = "%s.faiss" % faiss_path_prefix
    sql_rul = "sqlite:///%s.db" % faiss_path_prefix

    # CoreSearch Train Values
    max_seq_len_query = 64
    max_seq_len_passage = 180
    batch_size = 240
    processor_type = CoreSearchContextProcessor
    add_spatial_tokens = True
    similarity = "dot_product"
    # faiss_index_factory_str = "HNSW"
    faiss_index_factory_str = "Flat"

    passages_file = "data/resources/CoreSearch/clean/Dev_all_passages.json"
    # passages_file = "data/resources/CoreSearch/Tiny_passages.json"

    query_encode = "data/checkpoints/Retriever_SpanBERT_5it/1/query_encoder"
    passage_encode = "data/checkpoints/Retriever_SpanBERT_5it/1/passage_encoder"

    Path(faiss_dir).mkdir(exist_ok=False)
    print(f"creating document_store at-{faiss_path_prefix}, from documents-{passages_file}")
    print(f"query_encoder-{query_encode}, passage_encoder-{passage_encode}")
    faiss_index(passages_file, faiss_file_path, sql_rul, query_encode, passage_encode,
                max_seq_len_query, max_seq_len_passage, batch_size,
                similarity, processor_type, add_spatial_tokens, faiss_index_factory_str)


if __name__ == '__main__':
    main()
