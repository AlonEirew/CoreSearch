from src.utils import dpr_utils


def faiss_update(faiss_file_path, faiss_config_file, query_encode, passage_encode,
                 infer_tokenizer_classes, max_seq_len_query, max_seq_len_passage, batch_size, load_tokenizer):
    document_store, retriever = dpr_utils.load_faiss_dpr(faiss_file_path,
                                                         faiss_config_file,
                                                         query_encode,
                                                         passage_encode,
                                                         infer_tokenizer_classes,
                                                         max_seq_len_query,
                                                         max_seq_len_passage,
                                                         batch_size,
                                                         load_tokenizer)
    document_store.update_embeddings(retriever=retriever)
    document_store.save(faiss_file_path)
    print("Done!")


def main():
    faiss_path_prefix = "weces_index_for_bert_dpr/weces_dev_index"
    _faiss_file_path = "%s.faiss" % faiss_path_prefix
    _faiss_config_path = "%s.json" % faiss_path_prefix

    _infer_tokenizer_classes = True
    _max_seq_len_query = 50
    _max_seq_len_passage = 150
    _batch_size = 16

    _query_encode = "data/checkpoints/21022022_123254/model-13/query_encoder"
    _passage_encode = "data/checkpoints/21022022_123254/model-13/passage_encoder"
    _load_tokenizer = True

    faiss_update(_faiss_file_path, _faiss_config_path, _query_encode, _passage_encode,
                 _infer_tokenizer_classes, _max_seq_len_query, _max_seq_len_passage, _batch_size, _load_tokenizer)


if __name__ == '__main__':
    main()
