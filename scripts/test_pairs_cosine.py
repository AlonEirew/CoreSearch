import random
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm

from src.data_obj import Passage, Feat, Query, Cluster, QueryResult
from src.models.weces_retriever import WECESRetriever
from src.override_classes.wec_text_processor import WECSimilarityProcessor
from src.pipeline.run_haystack_pipeline import print_measurements
from src.utils import dpr_utils, io_utils, data_utils
from src.utils.data_utils import generate_index_batches
from src.utils.io_utils import load_checkpoint
from src.utils.tokenization import Tokenization

SPLIT = "Dev"
max_query_len = 50
max_pass_len = 150


def main():
    random.seed(1234)
    dev_examples_file = "data/resources/train/Dev_training_queries.json"
    # dev_examples_file = "data/resources/train/small_training_queries.json"
    dev_passages_file = "data/resources/train/Dev_training_passages.json"
    gold_cluster_file = "data/resources/WEC-ES/" + SPLIT + "_gold_clusters.json"
    model_file = "data/checkpoints/dev_spanbert_bm25_2it"
    topk = 50
    run_pipe_str = "retriever"
    batch_size = 40

    # query_examples: List[TrainExample] = io_utils.read_train_example_file(queries_file)
    # passage_examples: List[Passage] = io_utils.read_passages_file(passages_file)
    add_qbound = False

    ### NEED TO REMOVE THIS LINES\
    # for query in query_examples:
    #     query.context = query.bm25_query.split(" ")
    #
    # additional_pass = io_utils.read_passages_file_filtered("data/resources/WEC-ES/Dev_all_passages.json",
    #                                                        ['NEG_2078064', '11165', '122480'])
    # passage_examples.extend(additional_pass)

    model = load_dpr("data/checkpoints/dev_full_spanbert_bm25_2it", "indexes/spanbert_notft/dev_index")
    tokenization = model.processor.tokenization

    # tokenization = Tokenization(query_tok_file="SpanBERT/spanbert-base-cased", passage_tok_file="SpanBERT/spanbert-base-cased")
    # model = WECESRetriever("SpanBERT/spanbert-base-cased", "SpanBERT/spanbert-base-cased",
    #                        len(tokenization.query_tokenizer), "cuda")
    # model.to("cuda")
    ## END OF NEEDED LINES TO REMOVE

    # model, query_tokenizer, passage_tokenizer = load_checkpoint(model_file)
    # model.eval()
    # tokenization = Tokenization(query_tokenizer=query_tokenizer, passage_tokenizer=passage_tokenizer)
    print(f"Experiment using model={model_file}, query_file={dev_examples_file}, passage_file={dev_passages_file}")

    golds: List[Cluster] = io_utils.read_gold_file(gold_cluster_file)
    golds_arranged = data_utils.clusters_to_ids_list(gold_clusters=golds)

    dev_queries_feats = tokenization.generate_query_feats(dev_examples_file,
                                                          max_query_len,
                                                          add_qbound)
    dev_passages_feats = tokenization.generate_passage_feats(dev_passages_file,
                                                             max_pass_len)

    passage_dict: Dict[str, Passage] = {passage.feat_ref.id: passage.feat_ref for passage in dev_passages_feats}

    dev_queries_feats = random.sample(dev_queries_feats, k=20)
    total_queries = len(dev_queries_feats)

    dev_passages_ids, dev_passages_batches = generate_index_batches(list(dev_passages_feats), batch_size)
    all_dev_passages_encode = list()
    for batch_idx, batch in enumerate(tqdm(dev_passages_batches, "Encoding Passages")):
        pass_embeds = get_passage_embed(model, batch)
        all_dev_passages_encode.append(pass_embeds.detach().cpu())

    all_queries_pred = list()
    for query_index, query in enumerate(dev_queries_feats):
        # query = dev_queries_feats[10]
        query_predictions = run_top_pass(model, query, dev_passages_ids, all_dev_passages_encode, topk)
        results = list()
        for pass_id, pred in query_predictions:
            results.append(passage_dict[pass_id])
        query_result = QueryResult(query=query.feat_ref, results=results)
        all_queries_pred.append(query_result)

        print(f"queryId-{query.feat_ref.id}")
        print(f"queryMention-{query.feat_ref.mention}")
        for passage_id, prediction in query_predictions[:5]:
            print(passage_id + "(" + str(prediction) + ")-\"" + " ".join(passage_dict[passage_id].mention) + "\"")
        # if query_index == 0:
        #     break

    to_print = print_measurements(all_queries_pred, golds_arranged, run_pipe_str)
    join_result = "\n".join(to_print)
    print(join_result)
    print(f"Measured from-{total_queries}")
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


def run_top_pass(model, query: Feat, dev_passages_ids, all_dev_passages_encode, topk):
    query_encoded = get_query_embed(model, query)
    all_predictions: List[Tuple[str, float]] = list()
    for batch_idx, pass_batch_encoded in enumerate(tqdm(all_dev_passages_encode, "Evaluate")):
        batch_passage_ides = dev_passages_ids[batch_idx]
        predictions = WECESRetriever.predict_pairwise_cosine(query_encoded, pass_batch_encoded.to("cuda")).detach().cpu()
        for index in range(len(batch_passage_ides)):
            if batch_passage_ides[index] != query.feat_ref.id:
                all_predictions.append((batch_passage_ides[index], predictions[index].item()))

    final_predictions = sorted(all_predictions, key=lambda x: x[1], reverse=True)[:topk]
    return final_predictions


def get_query_embed(model, query):
    query_encoded, _ = model.query_encoder(
        torch.tensor(query.input_ids, device='cuda').view(1, -1),
        torch.tensor(query.segment_ids, device='cuda').view(1, -1),
        torch.tensor(query.input_mask, device='cuda').view(1, -1))
    return query_encoded


def get_passage_embed(model, passage_batch):
    input_ids, input_mask, segment_ids = passage_batch
    pos_pass_encoded, _ = model.passage_encoder(input_ids.cuda(),
                                                segment_ids.cuda(),
                                                input_mask.cuda())
    return pos_pass_encoded


if __name__ == "__main__":
    main()
