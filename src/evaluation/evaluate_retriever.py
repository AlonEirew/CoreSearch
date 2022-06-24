import copy
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
from src.override_classes.retriever.wec_bm25_processor import WECBM25Processor
from src.override_classes.retriever.wec_context_processor import WECContextProcessor
from src.override_classes.retriever.wec_dense import WECDensePassageRetriever
from src.pipeline.run_haystack_pipeline import print_measurements, generate_query_text
from src.utils import io_utils, data_utils
from src.utils.data_utils import generate_index_batches
from src.utils.io_utils import save_query_results

SPLIT = "Dev"


def main():
    random.seed(1234)
    examples_file = "data/resources/WEC-ES/train/" + SPLIT + "_queries.json"
    passages_file = "data/resources/WEC-ES/clean/" + SPLIT + "_all_passages.json"
    gold_cluster_file = "data/resources/WEC-ES/clean/" + SPLIT + "_gold_clusters.json"
    model_file = "data/checkpoints/Retriever_SpanBERT_notoks_5it/0"

    index_name = "file_indexes/" + SPLIT + "_Retriever_spanbert_notoks_5it0_top500"
    out_index_file = index_name + ".json"
    out_result_file = index_name + "_results.txt"

    max_query_len = 64
    max_pass_len = 180
    topk = 500
    batch_size = 240
    query_style = "context_no_toks"

    process_num = multiprocessing.cpu_count()

    if query_style == "bm25":
        processor_type = WECBM25Processor
        add_qbound = False
    elif query_style == "context":
        processor_type = WECContextProcessor
        add_qbound = True
    elif query_style == "context_no_toks":
        processor_type = WECContextProcessor
        add_qbound = False
    else:
        raise TypeError(f"No processor that support {query_style}")

    model = WECDensePassageRetriever(document_store=None, query_embedding_model=model_file + "/query_encoder",
                                     passage_embedding_model=model_file + "/passage_encoder",
                                     infer_tokenizer_classes=True,
                                     max_seq_len_query=max_query_len, max_seq_len_passage=max_pass_len,
                                     batch_size=16, use_gpu=True, embed_title=False,
                                     use_fast_tokenizers=False, processor_type=processor_type,
                                     add_special_tokens=add_qbound)

    print(f"Experiment using model={model_file}, query_file={examples_file}, passage_file={passages_file}")

    golds: List[Cluster] = io_utils.read_gold_file(gold_cluster_file)
    golds_arranged = data_utils.clusters_to_ids_list(gold_clusters=golds)

    query_examples: List[TrainExample] = io_utils.read_train_example_file(examples_file)
    passage_examples: List[Passage] = io_utils.read_passages_file(passages_file)
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

    save_query_results(all_queries_pred, out_index_file)

    to_print = list()
    to_print.append("Generating 50 examples:")
    query_pred_sample = random.sample(all_queries_pred, k=50)
    for query_res in query_pred_sample:
        to_print.append(f"queryId-{query_res.query.id}")
        to_print.append(f"queryMention-{query_res.query.mention}")
        to_print.append(f"queryContext-{query_res.query.context}")
        for passage_res in query_res.results[:5]:
            hit = True if passage_res.goldChain == query_res.query.goldChain else False
            to_print.append("\tHIT=(" + str(hit) + ")" + passage_res.id + " (" + str(passage_res.score) + ")-\"" + " ".join(passage_dict[passage_res.id].mention) + "\"")

    to_print.extend(print_measurements(all_queries_pred, golds_arranged, "retriever"))

    to_print.append(f"Measured from total of {total_queries} queries and {total_passages} passages")
    to_print.append(f"Using model-{model_file}")
    to_print.append(f"File: dev_queries={examples_file}, dev_passages={passages_file}, golds={gold_cluster_file}")
    to_print.append(f"Parameters: max_query_len={str(max_query_len)}, max_pass_len={str(max_pass_len)}, "
          f"topk={str(topk)}, add_qbound={str(add_qbound)}, query_style={query_style}")

    join_result = "\n".join(to_print)
    print("Saving report to-" + out_result_file)
    with open(out_result_file, 'w') as f:
        f.write(join_result)

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
        predictions = WECDensePassageRetriever.predict_pairwise_dot_product(query_encoded, pass_batch_encoded.cuda()).detach().cpu()
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
