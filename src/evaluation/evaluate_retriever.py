"""
Evaluate retriever script

Usage:
    evaluate_retriever.py [--note=<ExperimentSummary>] --query_filename=<QueryFile> --passages_filename=<PassagesFile> --gold_cluster_filename=<GoldClusterFile> --query_model=<QueryModel> --passage_model=<PassageModel> --out_index_file=<OutIndexFile> --out_results_file=<OutResultsFile> --num_processes=<NumProcesses> --add_special_tokens=<AddBoundToks> --max_seq_len_query=<MaxQuery> --max_seq_len_passage=<MaxPassage> --batch_size=<BatchSize> --top_k=<TopK>

Options:
    -h --help                                   Show this screen.
    --note=<ExperimentSummary>                  Brief note about the experiment
    --query_filename=<QueryFile>                CoreSearch query file
    --passages_filename=<PassagesFile>          CoreSearch passage file
    --gold_cluster_filename=<GoldClusterFile>   CoreSearch gold cluster file
    --query_model=<QueryModel>                  Query model to evaluate
    --passage_model=<PassageModel>              Passage model to evaluate
    --out_index_file=<OutIndexFile>             Where to save top-k passages index
    --out_results_file=<OutResultsFile>         Evaluation report
    --num_processes=<NumProcesses>              Number of processes to use for reading files
    --add_special_tokens=<AddBoundToks>         (true/false) whether to add span boundary tokens
    --max_seq_len_query=<MaxQuery>              Max query size
    --max_seq_len_passage=<MaxPassage>          Max passage size
    --batch_size=<BatchSize>                    Batch size
    --top_k=<TopK>                              The Top-K threshold for number of passages to return
"""

import os
import copy
import itertools
import math
import multiprocessing
import random
from multiprocessing import Pool
from typing import List, Dict, Tuple

import torch
from docopt import docopt
from haystack.modeling.utils import set_all_seeds
from tqdm import tqdm

from src.data_obj import Passage, Feat, Cluster, QueryResult, TrainExample
from src.override_classes.retriever.search_context_processor import CoreSearchContextProcessor
from src.override_classes.retriever.search_dense import CoreSearchDensePassageRetriever
from src.pipeline.run_e2e_pipeline import generate_query_text, print_measurements
from src.utils import io_utils, data_utils
from src.utils.data_utils import generate_index_batches
from src.utils.io_utils import save_query_results


def main():
    set_all_seeds(seed=42)
    query_filename = _arguments.get("--query_filename")
    passages_filename = _arguments.get("--passages_filename")
    gold_cluster_filename = _arguments.get("--gold_cluster_filename")
    query_model = _arguments.get("--query_model")
    passage_model = _arguments.get("--passage_model")
    out_index_file = _arguments.get("--out_index_file")
    out_results_file = _arguments.get("--out_results_file")

    add_special_tokens = True if _arguments.get("--add_special_tokens").lower() == 'true' else False
    max_seq_len_query = int(_arguments.get("--max_seq_len_query"))
    max_seq_len_passage = int(_arguments.get("--max_seq_len_passage"))
    top_k = int(_arguments.get("--top_k"))
    batch_size = int(_arguments.get("--batch_size"))
    num_processes = int(_arguments.get("--num_processes"))

    if not torch.cuda.is_available():
        raise EnvironmentError("Evaluation script require at least 1 GPU to run")

    if num_processes <= 0:
        num_processes = multiprocessing.cpu_count()

    if out_index_file is None or not os.path.exists(os.path.dirname(out_index_file)):
        raise ValueError(f"--out_index_file file folder-'{os.path.dirname(out_index_file)}' does not exist, create it first.")

    if out_results_file is None or not os.path.exists(os.path.dirname(out_results_file)):
        raise ValueError(f"--out_results_file file folder-'{os.path.dirname(out_results_file)}' does not exist, create it first.")

    model = CoreSearchDensePassageRetriever(document_store=None, query_embedding_model=query_model,
                                            passage_embedding_model=passage_model,
                                            infer_tokenizer_classes=True,
                                            max_seq_len_query=max_seq_len_query,
                                            max_seq_len_passage=max_seq_len_passage,
                                            batch_size=16, use_gpu=True, embed_title=False,
                                            use_fast_tokenizers=False, processor_type=CoreSearchContextProcessor,
                                            add_special_tokens=add_special_tokens)

    golds: List[Cluster] = io_utils.read_gold_file(gold_cluster_filename)
    golds_arranged = data_utils.clusters_to_ids_list(gold_clusters=golds)

    query_examples: List[TrainExample] = io_utils.read_train_example_file(query_filename)
    passage_examples: List[Passage] = io_utils.read_passages_file(passages_filename)
    passage_dict: Dict[str, Passage] = {passage.id: passage for passage in passage_examples}
    generate_query_text(passage_dict, query_examples=query_examples)
    dev_queries_feats = model.processor.generate_query_feats(query_examples)

    print(f"Reading passages using {num_processes} cpu's")
    chunk_size = math.ceil(len(passage_examples) / num_processes)
    passage_examples_chunks = [passage_examples[x:x + chunk_size] for x in range(0, len(passage_examples), chunk_size)]
    with Pool(num_processes) as pool:
        dev_passages_feats = pool.map(model.processor.generate_passage_feats, iterable=passage_examples_chunks)
        dev_passages_feats = list(itertools.chain.from_iterable(dev_passages_feats))

    # dev_queries_feats = random.sample(dev_queries_feats, k=20)
    total_queries = len(dev_queries_feats)
    total_passages = len(dev_passages_feats)

    dev_passages_ids, dev_passages_batches = generate_index_batches(list(dev_passages_feats), batch_size)
    net = torch.nn.DataParallel(model.passage_encoder, device_ids=[torch.cuda.device(i).idx for i in range(torch.cuda.device_count())])
    all_dev_passages_encode = extract_batch_list_embed(net, dev_passages_batches)

    all_queries_pred = list()
    for query_index, query in enumerate(tqdm(dev_queries_feats, "Evaluate Queries")):
        query_predictions = run_top_pass(model.query_encoder, query, dev_passages_ids, all_dev_passages_encode, top_k)
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
    to_print.append(f"File: dev_queries={query_filename}, dev_passages={passages_filename}, golds={gold_cluster_filename}")

    join_result = "\n".join(to_print)
    print("Saving report to-" + out_results_file)
    with open(out_results_file, 'w') as f:
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
        predictions = CoreSearchDensePassageRetriever.predict_pairwise_dot_product(query_encoded, pass_batch_encoded.cuda()).detach().cpu()
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
    _arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    print(_arguments)
    main()
