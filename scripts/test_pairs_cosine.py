from typing import List, Dict

import torch
from tqdm import tqdm

from src.data_obj import TrainExample, Passage, QueryFeat, PassageFeat
from src.models.weces_retriever import WECESRetriever
from src.utils import io_utils
from src.utils.io_utils import load_checkpoint
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
    passage_examples_dict: Dict[str, Passage] = {passage.id: passage for passage in passage_examples}

    # model = WECESRetriever(len(tokenization.tokenizer), "cuda")
    # tokenization = Tokenization()
    model, tokenizer = load_checkpoint("data/checkpoints/11032022_131710/model-0")
    model.eval()
    tokenization = Tokenization(tokenizer=tokenizer)

    # selected_query = query_examples[0]
    positive_win = 0
    negative_win = 0
    total_queries = len(query_examples)
    for index, selected_query in enumerate(query_examples):
        selected_query = query_examples[20]

        print(f"Query ID={selected_query.id}")
        print(f"Query Mention={selected_query.mention}")
        if experiment == "neg_pos":
            pairwise_pos, pairwise_neg = pos_neg_eval(model, tokenization, selected_query, passage_examples_dict)

            print(f"pos {index}={str(pairwise_pos.item())}")
            print(f"neg {index}={str(pairwise_neg.item())}")

            if pairwise_pos > pairwise_neg:
                positive_win += 1
            else:
                negative_win += 1

            # if index == 5:
            #     break
        elif experiment == "top_5":
            top_k = run_top_pass(model, tokenization, selected_query, passage_examples)
            print(f"top5_values={top_k[-5:]}")
            for passage_id, value in top_k[-5:]:
                print(passage_examples_dict[passage_id].mention)
            if index == 0:
                break

    print(f"total={total_queries}")
    print(f"positive_win={positive_win}")
    print(f"negative_win={negative_win}")

    print("Done!")


def run_top_pass(model, tokenization, selected_query, passage_examples: List[Passage]):
    query_encoded = get_query_embed(model, tokenization, selected_query)
    # random.shuffle(passage_examples)
    all_pairs = list()
    for passage in tqdm(passage_examples, desc="Running"):
        pass_encoded = get_passage_embed(model, tokenization, passage)
        all_pairs.append((passage.id, WECESRetriever.predict_pairwise(query_encoded, pass_encoded).item()))

    return sorted(all_pairs, key=lambda x: x[1])


def pos_neg_eval(model, tokenization, selected_query, passage_examples):
    positive_pass = passage_examples[selected_query.positive_examples[0]]
    negative_pass = passage_examples[selected_query.negative_examples[0]]

    query_encoded = get_query_embed(model, tokenization, selected_query)
    pos_pass_encoded = get_passage_embed(model, tokenization, positive_pass)
    neg_pass_encoded = get_passage_embed(model, tokenization, negative_pass)

    pairwise_pos = WECESRetriever.predict_pairwise(query_encoded, pos_pass_encoded)
    pairwise_neg = WECESRetriever.predict_pairwise(query_encoded, neg_pass_encoded)

    return pairwise_pos, pairwise_neg


def get_query_embed(model, tokenization, selected_query):
    Tokenization.add_query_bound(selected_query)
    query_feat: QueryFeat = tokenization.get_query_feat(selected_query, max_query_len)
    query_encoded, _ = model.query_encoder(
        torch.tensor(query_feat.query_input_ids, device=model.device).view(1, -1),
        torch.tensor(query_feat.query_segment_ids, device=model.device).view(1, -1),
        torch.tensor(query_feat.query_input_mask, device=model.device).view(1, -1),
        query_start=torch.tensor([query_feat.query_event_start], device=model.device),
        query_end=torch.tensor([query_feat.query_event_end], device=model.device),
        sample_size=1)
    return query_encoded


def get_passage_embed(model, tokenization, selected_passage):
    pass_feat: PassageFeat = tokenization.get_passage_feat(selected_passage, max_pass_len)
    pos_pass_encoded, _ = model.passage_encoder(
        torch.tensor(pass_feat.passage_input_ids, device=model.device).view(1, -1),
        torch.tensor(pass_feat.passage_segment_ids, device=model.device).view(1, -1),
        torch.tensor(pass_feat.passage_input_mask, device=model.device).view(1, -1))
    return pos_pass_encoded


if __name__ == "__main__":
    # _experiment = "neg_pos"
    _experiment = "top_5"
    main(_experiment)
