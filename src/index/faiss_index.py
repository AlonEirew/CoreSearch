from pathlib import Path
from typing import List, Dict

from haystack import Document
from tqdm import tqdm

from src.data_obj import Passage, PassageFeat
from src.utils import io_utils, dpr_utils
from src.utils.data_utils import generate_index_batches
from src.utils.tokenization import Tokenization


def faiss_index(passages_file,
                extract_embed,
                faiss_file_path,
                sql_rul,
                query_encode,
                passage_encode,
                infer_tokenizer_classes,
                max_seq_len_query,
                max_seq_len_passage,
                batch_size,
                load_tokenizer):

    document_store, retriever = dpr_utils.create_faiss_dpr(sql_rul,
                                                           query_encode,
                                                           passage_encode,
                                                           infer_tokenizer_classes,
                                                           max_seq_len_query,
                                                           max_seq_len_passage,
                                                           batch_size,
                                                           load_tokenizer)
    document_store.delete_documents()
    print("Writing document to FAISS index (may take a while)..")

    documents: List[Document] = io_utils.read_wec_to_haystack_doc_list(passages_file)
    doc_dict: Dict[str, Document] = {
        doc.id: doc for doc in documents
    }
    if extract_embed:
        tokenizer = Tokenization(tokenizer=retriever.passage_tokenizer)
        passages_examples: List[Passage] = io_utils.read_passages_file(passages_file)
        passages_feats: List[PassageFeat] = [
            tokenizer.get_passage_feat(passage, max_seq_len_passage) for passage in passages_examples
        ]
        ids, batches = generate_index_batches(passages_feats, batch_size)
        for i, batch in enumerate(tqdm(batches)):
            batch = tuple(t.cuda() for t in batch)
            pass_input_tensor, pass_segment_tensor, pass_input_mask_tensor = batch
            embeddings = retriever.passage_encoder(passage_input_ids=pass_input_tensor,
                                                   passage_segment_ids=pass_segment_tensor,
                                                   passage_attention_mask=pass_input_mask_tensor)[0]
            for j, _id in enumerate(ids[i]):
                doc_dict[_id].embedding = embeddings[j].detach().cpu().view(-1).numpy()

        document_store.write_documents(documents=documents)
    else:
        document_store.write_documents(documents=documents)
        document_store.update_embeddings(retriever=retriever)

    document_store.save(faiss_file_path)
    print("FAISS index creation done!")


def main():
    faiss_dir = "indexes/test"
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

    # passages_file = "data/resources/WEC-ES/Dev_all_passages.json"
    passages_file = "data/resources/WEC-ES/Tiny_passages.json"

    # query_encode = "bert-base-cased"
    # passage_encode = "bert-base-cased"
    # query_encode = "SpanBERT/spanbert-base-cased"
    # passage_encode = "SpanBERT/spanbert-base-cased"
    # query_encode = "facebook/dpr-question_encoder-multiset-base"
    # passage_encode = "facebook/dpr-ctx_encoder-multiset-base"
    # query_encode = "data/checkpoints/spanbert_2it/query_encoder"
    # passage_encode = "data/checkpoints/spanbert_2it/passage_encoder"
    query_encode = "data/checkpoints/21022022_123254/model-13/query_encoder"
    passage_encode = "data/checkpoints/21022022_123254/model-13/passage_encoder"

    load_tokenizer = True
    infer_tokenizer_classes = True
    extract_embed = True

    faiss_index(passages_file, extract_embed, faiss_file_path, sql_rul, query_encode, passage_encode,
                infer_tokenizer_classes, max_seq_len_query, max_seq_len_passage, batch_size, load_tokenizer)


if __name__ == '__main__':
    main()
