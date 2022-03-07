import logging
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AdamW

from src.models.coref_search_model import SimilarityModel
from src.models.wece_retriever import WECESRetriever
from src.utils.data_utils import generate_train_batches
from src.utils.evaluation import generate_sim_results, evaluate_retriever
from src.utils.io_utils import save_checkpoint
from src.utils.log_utils import create_logger
from src.utils.tokenization import Tokenization

logger = logging.getLogger("event-search")


def train():
    start_time = datetime.now()
    dt_string = start_time.strftime("%d%m%Y_%H%M%S")

    # train_examples_file = "data/resources/train/small_training_queries.json"
    # train_passages_file = "data/resources/train/Dev_training_passages.json"
    train_examples_file = "data/resources/train/Train_training_queries.json"
    train_passages_file = "data/resources/train/Train_training_passages.json"
    dev_examples_file = "data/resources/train/Dev_training_queries.json"
    dev_passages_file = "data/resources/train/Dev_training_passages.json"

    checkpoints_path = "data/checkpoints/" + dt_string
    Path(checkpoints_path).mkdir(parents=True)
    create_logger(log_file=checkpoints_path + "/log.txt")
    logger.info(f"{checkpoints_path}-folder created..")

    cpu_only = False
    epochs = 15
    train_negative_samples = 1
    dev_negative_samples = 5
    in_batch_samples = 10
    # train_batch_size = 3 * (2 + 1) in training for 2 negatives and 1 positive samples
    train_batch_size = in_batch_samples * (train_negative_samples + 1)
    dev_batch_size = in_batch_samples * (dev_negative_samples + 1)
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

    logger.info("Using device-" + device.type)
    logger.info("Number of available GPU's-" + str(n_gpu))

    if n_gpu > 0:
        torch.cuda.manual_seed_all(1234)

    tokenization = Tokenization()
    weces_retriever = WECESRetriever(len(tokenization.tokenizer), device)
    weces_retriever.to(device)
    if n_gpu > 1:
        weces_retriever = torch.nn.DataParallel(weces_retriever)

    similarity_method = SimilarityModel(device)

    optimizer = AdamW(weces_retriever.parameters(), lr=lr)

    train_search_feats = tokenization.generate_queries_feats(train_examples_file,
                                                             train_passages_file, max_query_length,
                                                             max_passage_length, train_negative_samples,
                                                             remove_qbound_tokens)

    dev_search_feats = tokenization.generate_queries_feats(dev_examples_file,
                                                           dev_passages_file, max_query_length,
                                                           max_passage_length, dev_negative_samples,
                                                           remove_qbound_tokens)

    train_batches = generate_train_batches(train_search_feats, train_batch_size)
    dev_batches = generate_train_batches(dev_search_feats, dev_batch_size)

    accum_loss = 0.0
    start_time = time.time()
    tot_steps = 0.0
    logger.info("Start training...")
    torch.autograd.set_detect_anomaly(False)
    for epoch in range(epochs):
        weces_retriever.train()
        random.shuffle(train_batches)
        batch_predictions, batch_golds = list(), list()
        for step, batch in enumerate(train_batches):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)

            passage_input_ids, query_input_ids, \
            passage_input_mask, query_input_mask, \
            passage_segment_ids, query_segment_ids, \
            passage_start_position, passage_end_position, \
            passage_end_bound, query_event_starts, query_event_ends = batch
            sim_loss, predictions = weces_retriever(passage_input_ids, query_input_ids,
                                                    passage_input_mask, query_input_mask,
                                                    passage_segment_ids, query_segment_ids,
                                                    query_event_starts, query_event_ends,
                                                    train_negative_samples + 1)

            if n_gpu > 1:
                sim_loss = sim_loss.mean()
            accum_loss += sim_loss.item()
            tot_steps += 1

            sim_loss.backward()
            # span_lost.backward()
            # sim_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_predictions.append(predictions.detach().cpu().numpy())
            batch_golds.append(np.zeros(len(predictions)))
            logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                epoch, step + 1, len(train_batches), time.time() - start_time, accum_loss / tot_steps))

        logger.info("Train-Similarity: accuracy={}".format(generate_sim_results(batch_golds, batch_predictions)))
        evaluate_retriever(weces_retriever, dev_batches, dev_negative_samples + 1, similarity_method, n_gpu)
        save_checkpoint(checkpoints_path, epoch, weces_retriever, tokenization, optimizer)


if __name__ == '__main__':
    train()
