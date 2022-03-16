from typing import List, Dict

import torch
from tqdm import tqdm

from src.data_obj import TrainExample, Passage, QueryFeat, PassageFeat
from src.models.weces_retriever import WECESRetriever
from src.override_classes.wec_text_processor import WECSimilarityProcessor
from src.utils import io_utils, dpr_utils
from src.utils.io_utils import load_checkpoint, replace_retriever_model
from src.utils.tokenization import Tokenization

SPLIT = "Dev"
max_query_len = 50
max_pass_len = 150


def main(experiment):
    queries_file = "data/resources/train/" + SPLIT + "_training_queries.json"
    # queries_file = "data/resources/train/small_training_queries.json"
    passages_file = "data/resources/train/" + SPLIT + "_training_passages.json"

    query_examples: List[TrainExample] = io_utils.read_train_example_file(queries_file)
    passage_examples: List[Passage] = io_utils.read_passages_file(passages_file)
    add_qbound = False

    ### NEED TO REMOVE THIS LINES
    additional_pass = io_utils.read_passages_file_filtered("data/resources/WEC-ES/Dev_all_passages.json",
                                                           ['NEG_2078064', '11165', '122480'])
    passage_examples.extend(additional_pass)

    model = load_dpr("data/checkpoints/dev_spanbert_2it", "indexes/multi_notft/dev_index")
    tokenization = model.processor.tokenization
    # tokenization = Tokenization(query_tok_file="bert-base-cased", passage_tok_file="bert-base-cased")
    ## END OF NEEDED LINES TO REMOVE

    passage_examples_dict: Dict[str, Passage] = {passage.id: passage for passage in passage_examples}

    # model, query_tokenizer, passage_tokenizer = load_checkpoint("data/checkpoints/15032022_144056/model-1")
    # model.eval()
    # tokenization = Tokenization(query_tokenizer=query_tokenizer, passage_tokenizer=passage_tokenizer)

    # selected_query = query_examples[0]
    positive_win = 0
    negative_win = 0
    total_queries = len(query_examples)
    for index, selected_query in enumerate(query_examples):
        selected_query = query_examples[20]

        print(f"Query ID={selected_query.id}")
        print(f"Query Mention={selected_query.mention}")
        if experiment == "neg_pos":
            pairwise_pos, pairwise_neg = pos_neg_eval(model, tokenization, selected_query,
                                                      passage_examples_dict, add_qbound)

            print(f"pos {index}={str(pairwise_pos.item())}")
            print(f"neg {index}={str(pairwise_neg.item())}")

            if pairwise_pos > pairwise_neg:
                positive_win += 1
            else:
                negative_win += 1

            # if index == 5:
            #     break
        elif experiment == "top_5":
            top_k = run_top_pass(model, tokenization, selected_query, passage_examples, add_qbound)
            print(f"top5_values={top_k[-5:]}")
            for passage_id, value in top_k[-5:]:
                print(passage_examples_dict[passage_id].mention)
            if index == 0:
                break

    print(f"total={total_queries}")
    print(f"positive_win={positive_win}")
    print(f"negative_win={negative_win}")

    print("Done!")


def load_dpr(model_dir, index_dir):
    faiss_index_prefix = index_dir
    faiss_index_file = faiss_index_prefix + ".faiss"
    faiss_config_file = faiss_index_prefix + ".json"
    model = dpr_utils.load_dpr(dpr_utils.load_faiss_doc_store(faiss_index_file, faiss_config_file),
                               model_dir + "/query_encoder",
                               model_dir + "/passage_encoder",
                               True,
                               max_query_len,
                               max_pass_len,
                               16,
                               False)

    model.processor = WECSimilarityProcessor(query_tokenizer=model.query_tokenizer,
                                             passage_tokenizer=model.passage_tokenizer,
                                             max_seq_len_passage=max_pass_len,
                                             max_seq_len_query=max_query_len,
                                             label_list=["hard_negative", "positive"],
                                             metric="text_similarity_metric",
                                             embed_title=False,
                                             num_hard_negatives=0,
                                             num_positives=1,
                                             tokenization=Tokenization(query_tokenizer=model.query_tokenizer,
                                                                       passage_tokenizer=model.passage_tokenizer))
    return model


def run_top_pass(model, tokenization, selected_query, passage_examples: List[Passage], add_qbound: bool):
    query_encoded = get_query_embed(model, tokenization, selected_query, add_qbound)
    all_pairs = list()
    for passage in tqdm(passage_examples, desc="Running"):
        pass_encoded = get_passage_embed(model, tokenization, passage)
        all_pairs.append((passage.id, WECESRetriever.predict_pairwise_cosine(query_encoded, pass_encoded).item()))

    return sorted(all_pairs, key=lambda x: x[1])


def pos_neg_eval(model, tokenization, selected_query, passage_examples, add_qbound):
    positive_pass = passage_examples[selected_query.positive_examples[0]]
    negative_pass = passage_examples[selected_query.negative_examples[0]]

    query_encoded = get_query_embed(model, tokenization, selected_query, add_qbound)
    pos_pass_encoded = get_passage_embed(model, tokenization, positive_pass)
    neg_pass_encoded = get_passage_embed(model, tokenization, negative_pass)

    pairwise_pos = WECESRetriever.predict_pairwise_cosine(query_encoded, pos_pass_encoded)
    pairwise_neg = WECESRetriever.predict_pairwise_cosine(query_encoded, neg_pass_encoded)

    return pairwise_pos, pairwise_neg


def get_query_embed(model, tokenization, selected_query, add_qbound):
    query_feat: QueryFeat = tokenization.get_query_feat(selected_query, max_query_len, add_qbound)
    query_encoded, _ = model.query_encoder(
        torch.tensor(query_feat.query_input_ids, device='cuda').view(1, -1),
        torch.tensor(query_feat.query_segment_ids, device='cuda').view(1, -1),
        torch.tensor(query_feat.query_input_mask, device='cuda').view(1, -1))
    return query_encoded


def get_passage_embed(model, tokenization, selected_passage):
    pass_feat: PassageFeat = tokenization.get_passage_feat(selected_passage, max_pass_len)
    pos_pass_encoded, _ = model.passage_encoder(
        torch.tensor(pass_feat.passage_input_ids, device='cuda').view(1, -1),
        torch.tensor(pass_feat.passage_segment_ids, device='cuda').view(1, -1),
        torch.tensor(pass_feat.passage_input_mask, device='cuda').view(1, -1))
    return pos_pass_encoded


if __name__ == "__main__":
    # _experiment = "neg_pos"
    _experiment = "top_5"
    main(_experiment)
