from pathlib import Path
from typing import List

from haystack import Document

from src.override_classes.wec_context_processor import WECContextProcessor
from src.utils import io_utils, dpr_utils


def faiss_index(passages_file,
                load_model,
                faiss_file_path,
                sql_rul,
                query_encode,
                passage_encode,
                infer_tokenizer_classes,
                max_seq_len_query,
                max_seq_len_passage,
                batch_size,
                similarity,
                processor_type,
                add_spatial_tokens,
                faiss_index_factory_str):
    documents: List[Document] = io_utils.read_wec_to_haystack_doc_list(passages_file)

    if load_model:
        document_store, retriever = dpr_utils.create_wec_faiss_dpr(sql_rul,
                                                                   query_encode,
                                                                   passage_encode,
                                                                   infer_tokenizer_classes,
                                                                   max_seq_len_query,
                                                                   max_seq_len_passage,
                                                                   batch_size,
                                                                   similarity,
                                                                   processor_type,
                                                                   add_spatial_tokens,
                                                                   faiss_index_factory_str)
    else:
        document_store, retriever = dpr_utils.create_faiss_dpr(sql_rul,
                                                               query_encode,
                                                               passage_encode,
                                                               infer_tokenizer_classes,
                                                               max_seq_len_query,
                                                               max_seq_len_passage,
                                                               batch_size,
                                                               similarity,
                                                               faiss_index_factory_str)
    document_store.delete_documents()
    print("Writing document to FAISS index (may take a while)..")
    document_store.write_documents(documents=documents)
    document_store.update_embeddings(retriever=retriever)
    document_store.save(faiss_file_path)
    print("FAISS index creation done!")


def main():
    faiss_dir = "indexes/spanbert_best"
    faiss_path_prefix = faiss_dir + "/dev_index"
    faiss_file_path = "%s.faiss" % faiss_path_prefix
    sql_rul = "sqlite:///%s.db" % faiss_path_prefix
    Path(faiss_dir).mkdir(exist_ok=False)

    # WEC Train Values
    max_seq_len_query = 64
    max_seq_len_passage = 180
    batch_size = 16
    processor_type = WECContextProcessor
    add_spatial_tokens = True
    similarity = "dot_product"
    # faiss_index_factory_str = "HNSW"
    faiss_index_factory_str = "Flat"

    passages_file = "data/resources/WEC-ES/Dev_all_passages.json"
    # passages_file = "data/resources/WEC-ES/Tiny_passages.json"

    query_encode = "data/checkpoints/dev_spanbert_hidden_cls_spatial_ctx_2it/query_encoder"
    passage_encode = "data/checkpoints/dev_spanbert_hidden_cls_spatial_ctx_2it/passage_encoder"

    infer_tokenizer_classes = True
    load_model = True

    print(f"creating document_store at-{faiss_path_prefix}, from documents-{passages_file}")
    print(f"query_encoder-{query_encode}, passage_encoder-{passage_encode}")
    faiss_index(passages_file, load_model, faiss_file_path, sql_rul, query_encode, passage_encode,
                infer_tokenizer_classes, max_seq_len_query, max_seq_len_passage, batch_size,
                similarity, processor_type, add_spatial_tokens, faiss_index_factory_str)


if __name__ == '__main__':
    main()
