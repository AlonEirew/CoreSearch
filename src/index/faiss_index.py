from pathlib import Path
from typing import List

from haystack import Document

from src.utils import io_utils, dpr_utils
from src.utils.io_utils import replace_loaded_retriever_model


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
                load_tokenizer,
                similarity="dot_product"):
    documents: List[Document] = io_utils.read_wec_to_haystack_doc_list(passages_file)
    document_store, retriever = dpr_utils.create_faiss_dpr(sql_rul,
                                                           query_encode,
                                                           passage_encode,
                                                           infer_tokenizer_classes,
                                                           max_seq_len_query,
                                                           max_seq_len_passage,
                                                           batch_size,
                                                           load_tokenizer,
                                                           similarity)
    document_store.delete_documents()
    print("Writing document to FAISS index (may take a while)..")
    if load_model:
        replace_loaded_retriever_model(retriever, load_model, max_seq_len_query, max_seq_len_passage)

    document_store.write_documents(documents=documents)
    document_store.update_embeddings(retriever=retriever)
    document_store.save(faiss_file_path)
    print("FAISS index creation done!")


def main():
    faiss_dir = "indexes/spanbert_ft_new"
    faiss_path_prefix = faiss_dir + "/dev_index"
    faiss_file_path = "%s.faiss" % faiss_path_prefix
    sql_rul = "sqlite:///%s.db" % faiss_path_prefix
    Path(faiss_dir).mkdir(exist_ok=False)

    # Default Values
    # max_seq_len_query = 64
    # max_seq_len_passage = 180
    batch_size = 16

    # WEC Train Values
    max_seq_len_query = 50
    max_seq_len_passage = 150

    passages_file = "data/resources/WEC-ES/Dev_all_passages.json"
    # passages_file = "data/resources/WEC-ES/Tiny_passages.json"

    # query_encode = "bert-base-cased"
    # passage_encode = "bert-base-cased"
    query_encode = "SpanBERT/spanbert-base-cased"
    passage_encode = "SpanBERT/spanbert-base-cased"
    # query_encode = "facebook/dpr-question_encoder-multiset-base"
    # passage_encode = "facebook/dpr-ctx_encoder-multiset-base"
    # query_encode = "data/checkpoints/spanbert_2it/query_encoder"
    # passage_encode = "data/checkpoints/spanbert_2it/passage_encoder"
    # query_encode = "data/checkpoints/08032022_092845/model-3/query_encoder"
    # passage_encode = "data/checkpoints/08032022_092845/model-3/passage_encoder"

    load_tokenizer = False
    infer_tokenizer_classes = True
    load_model = None #"data/checkpoints/mul_start_end_spacial_tokens/model-3"

    print(f"creating document_store at-{faiss_path_prefix}, from documents-{passages_file}")
    print(f"query_encoder-{query_encode}, passage_encoder-{passage_encode}")
    faiss_index(passages_file, load_model, faiss_file_path, sql_rul, query_encode, passage_encode,
                infer_tokenizer_classes, max_seq_len_query, max_seq_len_passage, batch_size, load_tokenizer)


if __name__ == '__main__':
    main()
