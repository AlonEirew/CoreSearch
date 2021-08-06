import copy
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW

from src.data_obj import SearchFeat
from src.model import WecEsModel
from src.utils import io_utils
from src.utils.data_utils import generate_batches
from src.utils.evaluation import evaluate
from src.utils.io_utils import save_checkpoint
from src.utils.log_utils import create_logger
from src.utils.tokenization import Tokenization

logger = logging.getLogger("event-search")


def generate_queries_feats(tokenizer: Tokenization,
                           query_file: str,
                           passages_file: str,
                           max_query_length: int,
                           max_passage_length: int,
                           negative_sample_size: int,
                           remove_qbound: bool = False) -> List[SearchFeat]:
    query_examples = io_utils.read_query_examples_file(query_file)
    passages_examples = io_utils.read_passages_file(passages_file)
    logger.info("Done loading examples file, queries-" + query_file + ", passages-" + passages_file)
    logger.info("Total examples loaded, queries=" + str(len(query_examples)) + ", passages=" + str(len(passages_examples)))
    logger.info("Starting to generate examples...")
    query_feats = dict()
    passage_feats = dict()
    search_feats = list()
    for query_obj in tqdm(query_examples.values(), "Loading Queries"):
        query_feat = tokenizer.get_query_feat(query_obj, max_query_length, max_passage_length, remove_qbound)
        query_feats[query_obj["id"]] = query_feat
        pos_passages = list()
        neg_passages = list()
        for pos_id in query_obj["positivePassagesIds"]:
            if pos_id not in passage_feats:
                passage_feats[pos_id] = tokenizer.get_passage_feat(passages_examples[pos_id], max_passage_length)
            pos_passages.append(passage_feats[pos_id])

        for neg_id in query_obj["negativePassagesIds"]:
            if neg_id not in passage_feats:
                passage_feats[neg_id] = tokenizer.get_passage_feat(passages_examples[neg_id], max_passage_length)
            passage_cpy = copy.copy(passage_feats[neg_id])
            passage_cpy.passage_event_start = passage_cpy.passage_event_end = 0
            neg_passages.append(passage_cpy)

        for pos_pass in pos_passages:
            search_feats.append(SearchFeat(query_feat, pos_pass, random.sample(neg_passages, negative_sample_size)))

    return search_feats


def extract_logits(outputs):
    softmax_starts = torch.softmax(outputs.start_logits, dim=1)
    softmax_ends = torch.softmax(outputs.end_logits, dim=1)
    start_idxs, end_idxs = torch.argmax(softmax_starts, dim=1), torch.argmax(softmax_ends, dim=1)
    print()


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
    create_logger(log_file=checkpoints_path + "/log.txt")
    logger.info(f"{checkpoints_path}-folder created..")

    cpu_only = False
    epochs = 15
    negative_sample_size = 2
    # batch_size = 3 * (2 + 1) in training for 2 negatives and 1 positive samples
    batch_size = 10 * (negative_sample_size + 1)
    lr = 1e-6
    remove_qbound_tokens = False
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

    train_search_feats = generate_queries_feats(tokenization, train_examples_file,
                                                train_passages_file, max_query_length,
                                                max_passage_length, negative_sample_size, remove_qbound_tokens)

    dev_search_feats = generate_queries_feats(tokenization, dev_examples_file,
                                              dev_passages_file, max_query_length,
                                              max_passage_length, negative_sample_size, remove_qbound_tokens)

    train_batches = generate_batches(train_search_feats, batch_size)
    dev_batches = generate_batches(dev_search_feats, batch_size)

    accum_loss = 0.0
    start_time = time.time()
    tot_steps = 0.0
    logger.info("Start training...")
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_batches)
        accum_outputs = list()
        for step, batch in enumerate(train_batches):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)

            passage_input_ids, query_input_ids, \
            passage_input_mask, query_input_mask, \
            passage_segment_ids, query_segment_ids, \
            passage_start_position, passage_end_position, \
            passage_end_bound, query_event_starts, query_event_ends = batch
            outputs = model(passage_input_ids, query_input_ids,
                            passage_input_mask, query_input_mask,
                            passage_segment_ids, query_segment_ids,
                            passage_start_position, passage_end_position)

            # start_idxs, ends_indxs = extract_logits(outputs)
            accum_outputs.append(outputs)

            loss = outputs.loss
            if n_gpu > 1:
                loss = loss.mean()
            accum_loss += loss.item()
            loss.backward()
            tot_steps += 1
            if step % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()

            logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                epoch, step + 1, len(train_batches), time.time() - start_time, accum_loss / tot_steps))

        evaluate(model, dev_batches, device)
        save_checkpoint(checkpoints_path, epoch, model, optimizer)


if __name__ == '__main__':
    train()
