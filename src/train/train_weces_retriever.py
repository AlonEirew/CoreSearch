import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import wandb
from transformers import AdamW

from src.data_obj import SearchFeat
from src.models.weces_retriever import WECESRetriever
from src.utils.data_utils import generate_train_batches
from src.utils.evaluation import generate_sim_results, evaluate_retriever
from src.utils.io_utils import save_checkpoint
from src.utils.log_utils import create_logger
from src.utils.tokenization import Tokenization

logger = logging.getLogger("event-search")


def train():
    start_time = datetime.now()
    dt_string = start_time.strftime("%d%m%Y_%H%M%S")

    train_examples_file = "data/resources/train/Dev_training_queries.json"
    train_passages_file = "data/resources/train/Dev_training_passages.json"
    # train_examples_file = "data/resources/train/Train_training_queries.json"
    # train_passages_file = "data/resources/train/Train_training_passages.json"
    dev_examples_file = "data/resources/train/Dev_training_queries.json"
    dev_passages_file = "data/resources/train/Dev_training_passages.json"

    query_model = "SpanBERT/spanbert-base-cased"
    passage_model = "SpanBERT/spanbert-base-cased"
    query_tokenizer_model = "SpanBERT/spanbert-base-cased"
    passage_tokenizer_model = "SpanBERT/spanbert-base-cased"
    logger.info(f"query_model:{query_model}, passage_model:{passage_model}, "
                f"query_tokenizer:{query_tokenizer_model}, passage_tokenizer:{passage_tokenizer_model}")

    checkpoints_path = "data/checkpoints/" + dt_string
    Path(checkpoints_path).mkdir(parents=True)
    create_logger(log_file=checkpoints_path + "/log.txt")
    logger.info(f"{checkpoints_path}-folder created..")

    add_qbound_tokens = False
    cpu_only = False
    epochs = 3
    batch_size = 16
    train_negative_samples = 1
    dev_negative_samples = 1
    lr = 1e-6
    max_query_length = 50
    max_passage_length = 150
    assert (max_query_length + max_passage_length + 3) <= 512
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu_only else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1243)

    wandb.config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "negative_samples": train_negative_samples
    }

    logger.info("Using device-" + device.type)
    logger.info("Number of available GPU's-" + str(n_gpu))

    if n_gpu > 0:
        torch.cuda.manual_seed_all(1234)

    tokenization = Tokenization(query_tok_file=query_tokenizer_model, passage_tok_file=passage_tokenizer_model)
    weces_retriever = WECESRetriever(query_model, passage_model, len(tokenization.query_tokenizer), device)
    weces_retriever.to(device)
    if n_gpu > 1:
        weces_retriever = torch.nn.DataParallel(weces_retriever)

    optimizer = AdamW(weces_retriever.parameters(), lr=lr)

    train_search_feats = tokenization.generate_train_search_feats(
        train_examples_file,
        train_passages_file,
        max_query_length,
        max_passage_length,
        train_negative_samples,
        add_qbound_tokens)

    dev_search_feats = tokenization.generate_train_search_feats(
        dev_examples_file,
        dev_passages_file,
        max_query_length,
        max_passage_length,
        dev_negative_samples,
        add_qbound_tokens)

    accum_loss = 0.0
    start_time = time.time()
    tot_steps = 0.0
    logger.info("Start training...")
    torch.autograd.set_detect_anomaly(False)

    random.shuffle(train_search_feats)
    random.shuffle(dev_search_feats)
    train_batches = generate_train_batches(train_search_feats,
                                           train_negative_samples,
                                           batch_size,
                                           max_passage_length)
    dev_batches = generate_train_batches(dev_search_feats,
                                         dev_negative_samples,
                                         batch_size,
                                         max_passage_length)
    for epoch in range(epochs):
        weces_retriever.train()
        batch_predictions, batch_golds = list(), list()
        for step, batch in enumerate(train_batches):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)

            passage_input_ids, query_input_ids, \
            passage_input_mask, query_input_mask, \
            passage_segment_ids, query_segment_ids, \
            passage_start_position, passage_end_position, \
            passage_end_bound, query_event_starts, query_event_ends = batch
            loss, softmax_scores, passage_pos_indices = weces_retriever(passage_input_ids,
                                                                     query_input_ids,
                                                                     passage_input_mask,
                                                                     query_input_mask,
                                                                     passage_segment_ids,
                                                                     query_segment_ids,
                                                                     query_start=query_event_starts,
                                                                     query_end=query_event_ends,
                                                                     sample_size=train_negative_samples + 1)

            # if n_gpu > 1:
            #     loss = loss.mean()
            accum_loss += loss.item()
            tot_steps += 1

            loss.backward()
            # span_lost.backward()
            # sim_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _ ,torch_max = torch.max(softmax_scores.detach().cpu(), 1)
            batch_predictions.append(torch_max.numpy())
            batch_golds.append(passage_pos_indices.detach().cpu().numpy())
            logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                epoch, step + 1, len(train_batches), time.time() - start_time, accum_loss / tot_steps))

        wandb.log({"loss": accum_loss / tot_steps, })
        train_accuracy = generate_sim_results(batch_golds, batch_predictions)
        logger.info("Train-Similarity: accuracy={}".format(train_accuracy))
        # dev_accuracy = evaluate_retriever(weces_retriever, dev_batches, dev_negative_samples + 1, n_gpu)
        # wandb.log({"train-accurach": train_accuracy, "dev-accuracy": dev_accuracy, "loss": accum_loss / tot_steps})
        wandb.watch(weces_retriever)
        save_checkpoint(checkpoints_path, epoch, weces_retriever, tokenization, optimizer)

    wandb.finish()


def show_heat_map(weces_retriever: WECESRetriever, tokenization: Tokenization, train_search_feats: List[SearchFeat],
                  step: str):
    samples = train_search_feats[20:21]
    with torch.no_grad():
        for sample in samples:
            query_input = sample.query
            query_ref = sample.query.feat_ref
            pos_passage = sample.positive_passage
            neg_passage = sample.negative_passages[1]
            query_encode = weces_retriever.query_encoder.segment_encode(
                torch.tensor(query_input.input_ids, device=weces_retriever.device).view(1, -1),
                torch.tensor(query_input.segment_ids, device=weces_retriever.device).view(1, -1),
                torch.tensor(query_input.input_mask, device=weces_retriever.device).view(1, -1))

            pos_pass_encode = weces_retriever.passage_encoder.segment_encode(
                torch.tensor(pos_passage.input_ids, device=weces_retriever.device).view(1, -1),
                torch.tensor(pos_passage.segment_ids, device=weces_retriever.device).view(1, -1),
                torch.tensor(pos_passage.input_mask, device=weces_retriever.device).view(1, -1))
            query_tokenized = tokenization.query_tokenizer.convert_ids_to_tokens(query_input.input_ids)
            query_labels = [query_tokenized[0]] + query_tokenized[query_input.query_event_start:query_input.query_event_end+1]
            passage_tokenized = tokenization.passage_tokenizer.convert_ids_to_tokens(pos_passage.input_ids)
            passage_labels = [passage_tokenized[0]] + passage_tokenized[pos_passage.passage_event_start:pos_passage.passage_event_end+1]
            label = " ".join(query_ref.mention) + " .VS. " + " ".join(pos_passage.feat_ref.mention)
            query_encode = query_encode[0].squeeze()
            query_final_encode = torch.cat((query_encode[0].view(1, -1), query_encode[query_input.query_event_start:query_input.query_event_end+1]))
            pos_pass_encode = pos_pass_encode[0].squeeze()
            pos_pass_final_encode = torch.cat((pos_pass_encode[0].view(1, -1), pos_pass_encode[pos_passage.passage_event_start:pos_passage.passage_event_end+1]))
            matrix_values = query_final_encode @ pos_pass_final_encode.T
            # matrix_values = torch.cosine_similarity(query_final_encode, pos_pass_final_encode)
            wandb.log({label + "-" + step: wandb.plots.HeatMap(passage_labels, query_labels, matrix_values.cpu(), show_text=False)})


if __name__ == '__main__':
    wandb.init(project="weces", entity="eirew")
    random.seed(1234)
    train()
