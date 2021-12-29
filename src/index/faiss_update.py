from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever


def faiss_update(faiss_file_path, faiss_config_path, retriever_model):
    document_store = FAISSDocumentStore.load(index_path=faiss_file_path,
                                             config_path=faiss_config_path)

    retriever = DensePassageRetriever.load(load_dir=retriever_model,
                                           document_store=document_store,
                                           infer_tokenizer_classes=True)

    document_store.update_embeddings(retriever=retriever)
    document_store.save(faiss_file_path)
    print("Done!")


def main():
    faiss_path_prefix = "weces_index_for_bert_dpr/weces_dev_index"
    _faiss_file_path = "%s.faiss" % faiss_path_prefix
    _faiss_config_path = "%s.json" % faiss_path_prefix
    _retriever_model = "data/checkpoints/dpr_bert_1it"
    faiss_update(_faiss_file_path, _faiss_config_path, _retriever_model)


if __name__ == '__main__':
    main()
