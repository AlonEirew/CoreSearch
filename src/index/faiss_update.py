from src.override_classes.retriever.search_context_processor import CoreSearchContextProcessor
from src.override_classes.retriever.search_dense import CoreSearchDensePassageRetriever
from src.utils.dpr_utils import load_faiss_doc_store


def faiss_update(faiss_file_path, faiss_config_file, query_encode, passage_encode,
                 max_seq_len_query, max_seq_len_passage, batch_size):
    document_store = load_faiss_doc_store(faiss_file_path, faiss_config_file)
    retriever = CoreSearchDensePassageRetriever(document_store=document_store,
                                                query_embedding_model=query_encode,
                                                passage_embedding_model=passage_encode,
                                                infer_tokenizer_classes=True,
                                                max_seq_len_query=max_seq_len_query,
                                                max_seq_len_passage=max_seq_len_passage,
                                                batch_size=batch_size, use_gpu=True, embed_title=False,
                                                use_fast_tokenizers=False,
                                                processor_type=CoreSearchContextProcessor,
                                                add_special_tokens=True)

    document_store.update_embeddings(retriever=retriever)
    document_store.save(faiss_file_path)
    print("Done!")


def main():
    faiss_path_prefix = "faiss_indexes/best_index/dev_test_index"
    _faiss_file_path = "%s.faiss" % faiss_path_prefix
    _faiss_config_path = "%s.json" % faiss_path_prefix

    _max_seq_len_query = 64
    _max_seq_len_passage = 180
    _batch_size = 2500

    _query_encode = "data/checkpoints/Retriever_SpanBERT/0/query_encoder"
    _passage_encode = "data/checkpoints/Retriever_SpanBERT/0/passage_encoder"

    print(f"Preparing to update index: {faiss_path_prefix}, with query_model:{_query_encode}, passage_model:{_passage_encode}")
    faiss_update(_faiss_file_path, _faiss_config_path, _query_encode, _passage_encode,
                 _max_seq_len_query, _max_seq_len_passage, _batch_size)


if __name__ == '__main__':
    main()
