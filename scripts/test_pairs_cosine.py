import copy
import itertools
import math
import multiprocessing
import random
from multiprocessing import Pool
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm

from src.data_obj import Passage, Feat, Cluster, QueryResult, TrainExample
from src.models.weces_retriever import WECESRetriever

from src.override_classes.wec_bm25_processor import WECBM25Processor
from src.override_classes.wec_context_processor import WECContextProcessor
from src.pipeline.run_haystack_pipeline import print_measurements, generate_query_text
from src.utils import dpr_utils, io_utils, data_utils
from src.utils.data_utils import generate_index_batches

SPLIT = "Dev"


def main():
    random.seed(1234)
    dev_examples_file = "data/resources/train/Dev_training_queries.json"
    # dev_examples_file = "data/resources/train/small_training_queries.json"
    dev_passages_file = "data/resources/WEC-ES/Dev_all_passages.json"
    # dev_passages_file = "data/resources/train/Dev_training_passages.json"
    gold_cluster_file = "data/resources/WEC-ES/" + SPLIT + "_gold_clusters.json"
    # model_file = "data/checkpoints/dev_spanbert_bm25_2it"
    model_file = "data/checkpoints/dev_spanbert_hidden_cls_full_ctx_2it"

    max_query_len = 64
    max_pass_len = 180
    topk = 100
    run_pipe_str = "retriever"
    batch_size = 240
    process_num = multiprocessing.cpu_count()

    add_qbound = False
    query_style = "context"

    if query_style == "bm25":
        processor_type = WECBM25Processor
    elif query_style == "context":
        processor_type = WECContextProcessor
    else:
        raise TypeError(f"No processor that support {query_style}")

    ### NEED TO REMOVE THIS LINES?
    #
    # additional_pass = io_utils.read_passages_file_filtered("data/resources/WEC-ES/Dev_all_passages.json",
    #                                                        ['NEG_2078064', '11165', '122480'])
    # passage_examples.extend(additional_pass)
    faiss_index_prefix = "indexes/spanbert_notft/dev_index"
    faiss_index_file = faiss_index_prefix + ".faiss"
    faiss_config_file = faiss_index_prefix + ".json"
    _, model = dpr_utils.load_wec_faiss_dpr(faiss_index_file,
                                            faiss_config_file,
                                            model_file + "/query_encoder",
                                            model_file + "/passage_encoder",
                                            True,
                                            max_query_len,
                                            max_pass_len,
                                            16,
                                            processor_type,
                                            add_qbound)

    # tokenization = Tokenization(query_tok_file="SpanBERT/spanbert-base-cased", passage_tok_file="SpanBERT/spanbert-base-cased")
    # model = WECESRetriever("SpanBERT/spanbert-base-cased", "SpanBERT/spanbert-base-cased",
    #                        len(tokenization.query_tokenizer), "cuda")
    # model.to("cuda")
    ## END OF NEEDED LINES TO REMOVE?

    # model, query_tokenizer, passage_tokenizer = load_checkpoint(model_file)
    # model.eval()
    # tokenization = Tokenization(query_tokenizer=query_tokenizer, passage_tokenizer=passage_tokenizer)
    print(f"Experiment using model={model_file}, query_file={dev_examples_file}, passage_file={dev_passages_file}")

    golds: List[Cluster] = io_utils.read_gold_file(gold_cluster_file)
    golds_arranged = data_utils.clusters_to_ids_list(gold_clusters=golds)

    query_examples: List[TrainExample] = io_utils.read_train_example_file(dev_examples_file)
    passage_examples: List[Passage] = io_utils.read_passages_file(dev_passages_file)
    passage_dict: Dict[str, Passage] = {passage.id: passage for passage in passage_examples}
    generate_query_text(passage_dict, query_examples=query_examples, query_method=query_style)
    dev_queries_feats = model.processor.generate_query_feats(query_examples)

    print(f"Reading passages using {process_num} cpu's")
    chunk_size = math.ceil(len(passage_examples) / process_num)
    passage_examples_chunks = [passage_examples[x:x + chunk_size] for x in range(0, len(passage_examples), chunk_size)]
    with Pool(process_num) as pool:
        dev_passages_feats = pool.map(model.processor.generate_passage_feats, iterable=passage_examples_chunks)
        dev_passages_feats = list(itertools.chain.from_iterable(dev_passages_feats))

    # dev_queries_feats = random.sample(dev_queries_feats, k=20)
    total_queries = len(dev_queries_feats)
    total_passages = len(dev_passages_feats)

    dev_passages_ids, dev_passages_batches = generate_index_batches(list(dev_passages_feats), batch_size)
    net = torch.nn.DataParallel(model.passage_encoder, device_ids=[0, 1, 2, 3])
    all_dev_passages_encode = extract_batch_list_embed(net, dev_passages_batches)

    all_queries_pred = list()
    for query_index, query in enumerate(tqdm(dev_queries_feats, "Evaluate Queries")):
        query_predictions = run_top_pass(model.query_encoder, query, dev_passages_ids, all_dev_passages_encode, topk)
        results = list()
        for pass_id, pred in query_predictions:
            res_pass = copy.deepcopy(passage_dict[pass_id])
            res_pass.score = pred
            results.append(res_pass)
        query_result = QueryResult(query=query.feat_ref, results=results)
        all_queries_pred.append(query_result)

    print("Generating 50 examples:")
    query_pred_sample = random.sample(all_queries_pred, k=50)
    for query_res in query_pred_sample:
        print(f"queryId-{query_res.query.id}")
        print(f"queryMention-{query_res.query.mention}")
        print(f"queryContext-{query_res.query.context}")
        for passage_res in query_res.results[:5]:
            hit = True if passage_res.goldChain == query_res.query.goldChain else False
            print("\tHIT=(" + str(hit) + ")" + passage_res.id + " (" + str(passage_res.score) + ")-\"" + " ".join(passage_dict[passage_res.id].mention) + "\"")

    to_print = print_measurements(all_queries_pred, golds_arranged, run_pipe_str)
    join_result = "\n".join(to_print)
    print(join_result)
    print(f"Measured from total of {total_queries} queries and {total_passages} passages")
    print(f"Using model-{model_file}")
    print(f"File: dev_queries={dev_examples_file}, dev_passages={dev_passages_file}, golds={gold_cluster_file}")
    print(f"Parameters: max_query_len={str(max_query_len)}, max_pass_len={str(max_pass_len)}, "
          f"topk={str(topk)}, add_qbound={str(add_qbound)}, query_style={query_style}")
    print("Done!")


def extract_batch_list_embed(pass_encoder, dev_passages_batches):
    all_batch_embeds = list()
    for batch_idx, batch in enumerate(tqdm(dev_passages_batches, "Encoding Passages")):
        all_batch_embeds.append(get_passage_embed(pass_encoder, batch).detach().cpu())
    return all_batch_embeds


def run_top_pass(model, query: Feat, dev_passages_ids, all_dev_passages_encode, topk):
    query_encoded = get_query_embed(model, query)
    all_predictions: List[Tuple[str, float]] = list()
    for batch_idx, pass_batch_encoded in enumerate(all_dev_passages_encode):
        batch_passage_ides = dev_passages_ids[batch_idx]
        predictions = WECESRetriever.predict_pairwise_dot_product(query_encoded, pass_batch_encoded.cuda()).detach().cpu()
        for index in range(len(batch_passage_ides)):
            if batch_passage_ides[index] != query.feat_ref.id:
                all_predictions.append((batch_passage_ides[index], predictions[index].item()))

    final_predictions = sorted(all_predictions, key=lambda x: x[1], reverse=True)[:topk]
    return final_predictions


def get_query_embed(model, query):
    query_encoded, _ = model(
        torch.tensor(query.input_ids, device='cuda').view(1, -1),
        torch.tensor(query.segment_ids, device='cuda').view(1, -1),
        torch.tensor(query.input_mask, device='cuda').view(1, -1))
    return query_encoded


def get_passage_embed(model, passage_batch):
    input_ids, input_mask, segment_ids = passage_batch
    pos_pass_encoded, _ = model(input_ids.cuda(),
                                segment_ids.cuda(),
                                input_mask.cuda())
    return pos_pass_encoded


if __name__ == "__main__":
    main()
