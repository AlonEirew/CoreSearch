import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW

from src.data_obj import SearchFeat, PassageFeat, QueryFeat
from src.model import WecEsModel
from src.utils import io_utils
from src.utils.evaluation import evaluate
from src.utils.io_utils import save_checkpoint
from src.utils.search_dataset import SearchFeatDataset
from src.utils.tokenization import Tokenization


def generate_queries_feats(tokenizer: Tokenization,
                           query_file: str,
                           passages_file: str,
                           max_query_length: int,
                           max_passage_length: int,
                           remove_qbound: bool = False) -> Tuple[Dict[int, QueryFeat], Dict[int, PassageFeat], List[SearchFeat]]:
    query_examples = io_utils.read_query_examples_file(query_file)
    passages_examples = io_utils.read_passages_file(passages_file)
    print("Done loading examples file, queries-" + query_file + ", passages-" + passages_file)
    print("Total examples loaded, queries=" + str(len(query_examples)) + ", passages=" + str(len(passages_examples)))
    print("Starting to generate examples...")
    query_feats = dict()
    passage_feats = dict()
    search_feats = list()
    for query_obj in tqdm(query_examples.values(), "Loading Queries"):
        query_feats[query_obj["id"]] = tokenizer.get_query_feat(query_obj, max_query_length, max_passage_length, remove_qbound)
        pos_passages_ids = list()
        neg_passages_ids = list()
        for pos_id in query_obj["positivePassagesIds"]:
            if pos_id not in passage_feats:
                pos_pass_feat = tokenizer.get_passage_feat(passages_examples[pos_id], max_passage_length)
                passage_feats[pos_id] = pos_pass_feat
            pos_passages_ids.append(pos_id)

        for neg_id in query_obj["negativePassagesIds"]:
            if neg_id not in passage_feats:
                pos_pass_feat = tokenizer.get_passage_feat(passages_examples[neg_id], max_passage_length)
                passage_feats[neg_id] = pos_pass_feat
            neg_passages_ids.append(neg_id)

        for pos_id in pos_passages_ids:
            search_feats.append(SearchFeat(query_obj["id"], pos_id, neg_passages_ids))

    return query_feats, passage_feats, search_feats


def train():
    start_time = datetime.now()
    dt_string = start_time.strftime("%d%m%Y_%H%M%S")

    train_examples_file = "resources/train/wec_es_train_qsent_small.json"
    train_passages_file = "resources/train/wec_es_Train_passages_segment.json"
    dev_examples_file = train_examples_file#"resources/train/wec_es_Dev_qsent_psegment_examples.json"
    dev_passages_file = train_passages_file#"resources/train/wec_es_Dev_passages_segment.json"

    tokenizer_path = "checkpoints/" + dt_string + "/tokenizer"
    checkpoints_path = "checkpoints/" + dt_string
    Path(checkpoints_path).mkdir(parents=True)
    Path(tokenizer_path).mkdir()
    print(f"{checkpoints_path}-folder created..")

    cpu_only = False
    epochs = 15
    batch_size = 32
    lr = 1e-6
    remove_qbound_tokens = False
    # hidden_size = 500
    max_query_length = 50
    max_passage_length = 150
    assert (max_query_length + max_passage_length + 3) <= 512
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu_only else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1243)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(1234)

    tokenization = Tokenization()
    tokenization.tokenizer.save_pretrained(tokenizer_path)
    model = WecEsModel(len(tokenization.tokenizer))
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=lr)

    train_queries, train_passages, train_search_feats = generate_queries_feats(tokenization,
                                                                               train_examples_file,
                                                                               train_passages_file,
                                                                               max_query_length,
                                                                               max_passage_length,
                                                                               remove_qbound_tokens)

    dev_queries, dev_passages, dev_search_feats = generate_queries_feats(tokenization,
                                                                         dev_examples_file,
                                                                         dev_passages_file,
                                                                         max_query_length,
                                                                         max_passage_length,
                                                                         remove_qbound_tokens)

    train_batches = SearchFeatDataset.generate_search_batches(train_search_feats, batch_size)
    dev_batches = SearchFeatDataset.generate_search_batches(dev_search_feats, batch_size)

    accum_loss = 0.0
    start_time = time.time()
    tot_steps = 0.0
    print("Start training...")
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_batches)
        for step, batch in enumerate(train_batches):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)
            query_id_list, positive_id_list, negative_ids_list = batch
            assert query_id_list.shape[0] == positive_id_list.shape[0] == negative_ids_list[0].shape[0]
            for i in range(len(query_id_list)):
                query_id, positive_id, negative_ids = query_id_list[i].item(), positive_id_list[i].item(), [neg[i].item() for neg in negative_ids_list]
                passage_input_ids, query_input_ids, \
                passage_input_mask, query_input_mask, \
                passage_segment_ids, query_segment_ids, \
                passage_start_position, passage_end_position = SearchFeatDataset.generate_span_feats(
                    train_queries, train_passages, query_id, positive_id, negative_ids, device)
                outputs = model(passage_input_ids, query_input_ids,
                                passage_input_mask, query_input_mask,
                                passage_segment_ids, query_segment_ids,
                                passage_start_position, passage_end_position)

                loss = outputs.loss
                if n_gpu > 1:
                    loss = loss.mean()
                accum_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tot_steps += 1

            print('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                epoch, step + 1, len(train_batches), time.time() - start_time, accum_loss / tot_steps))

        evaluate(model, dev_batches, device)
        save_checkpoint(checkpoints_path, epoch, model, optimizer)


if __name__ == '__main__':
    train()
