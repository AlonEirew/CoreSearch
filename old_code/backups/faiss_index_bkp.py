from pathlib import Path
from typing import List, Dict

import torch
from haystack import Document

from src.coref_search_model import SpanPredAuxiliary
from src.utils.dpr_utils import create_default_faiss_doc_store
from tqdm import tqdm
from transformers import BertTokenizer

from src.data_obj import Passage, PassageFeat
from src.utils import io_utils, dpr_utils
from src.utils.data_utils import generate_index_batches
from src.utils.io_utils import load_model_bkp
from src.utils.tokenization import Tokenization


def faiss_orig_index(passages_file,
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

    documents: List[Document] = io_utils.read_coresearch_to_haystack_doc_list(passages_file)
    document_store.write_documents(documents=documents)
    document_store.update_embeddings(retriever=retriever)

    document_store.save(faiss_file_path)
    print("FAISS index creation done!")


def faiss_mymodel_index(passages_file, sql_url, passage_model_file, max_seq_len_passage, batch_size, device):
    document_store = create_default_faiss_doc_store(sql_url)
    passage_encoder = load_model_bkp(passage_model_file)
    passage_encoder.eval()
    if device == "gpu":
        passage_encoder.cuda()

    passage_tokenizer = BertTokenizer.from_pretrained(passage_model_file)

    documents: List[Document] = io_utils.read_coresearch_to_haystack_doc_list(passages_file)
    doc_dict: Dict[str, Document] = {
        doc.id: doc for doc in documents
    }
    tokenizer = Tokenization(query_tokenizer=passage_tokenizer)
    passages_examples: List[Passage] = io_utils.read_passages_file(passages_file)
    passages_feats: List[PassageFeat] = [
        tokenizer.get_passage_feat(passage) for passage in passages_examples
    ]

    ids, batches = generate_index_batches(passages_feats, batch_size)
    head_mask = [None] * passage_encoder.config.num_hidden_layers
    for i, batch in enumerate(tqdm(batches)):
        batch = tuple(t.cuda() for t in batch)
        pass_input_tensor, pass_segment_tensor, pass_input_mask_tensor = batch
        embeddings = SpanPredAuxiliary.segment_encode(passage_encoder,
                                                      pass_input_tensor,
                                                      pass_segment_tensor,
                                                      pass_input_mask_tensor,
                                                      head_mask)[0]
        for j, _id in enumerate(ids[i]):
            doc_dict[_id].embedding = embeddings[j].detach().cpu().numpy()

    document_store.write_documents(documents=documents)


def main():
    faiss_dir = "indexes/test"
    faiss_path_prefix = faiss_dir + "/dev_index"
    faiss_file_path = "%s.faiss" % faiss_path_prefix
    sql_url = "sqlite:///%s.db" % faiss_path_prefix
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
    extract_embed = False
    device = "gpu"

    if extract_embed:
        faiss_mymodel_index(passages_file, sql_url, passage_encode, max_seq_len_passage, batch_size, device)
    else:
        faiss_orig_index(passages_file, faiss_file_path, sql_url, query_encode, passage_encode,
                         infer_tokenizer_classes, max_seq_len_query, max_seq_len_passage, batch_size, load_tokenizer)


if __name__ == '__main__':
    main()
