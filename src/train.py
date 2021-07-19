import random
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW

from src import io_utils
from src.data.input_feature import InputFeature
from src.model import WecEsModel


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1
    batch_size = 32
    lr = 1e-6
    hidden_size = 500
    max_query_length = 100
    max_passage_length = 300
    assert (max_query_length + max_passage_length + 3) <= 512
    n_gpu = torch.cuda.device_count()
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1243)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(1234)

    model = WecEsModel(hidden_size)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    train_data = read_and_gen_features(model, max_query_length, max_passage_length)
    train_batches = generate_data_loader_batches(train_data, batch_size)

    accum_loss = 0.0
    start_time = time.time()
    tot_steps = 0.0
    print("Start training...")
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        for step, batch in enumerate(train_batches):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, start_position, end_position = batch
            loss = model(input_ids, input_mask, segment_ids, start_position, end_position)
            accum_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tot_steps += 1
            print('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                epoch, step + 1, len(train_data), time.time() - start_time, accum_loss / tot_steps))

        evaluate()


def evaluate():
    pass


def read_and_gen_features(model, max_query_length, max_passage_length) -> List[InputFeature]:
    # train_gold = utils.read_gold_file("resources/gold/Train_query_to_relevant_passages.tsv")
    # Read the json format of the queries examples with positive and negative passages ids

    # list of (query_id, passage_id, pass_ment_bound, gold) where pass_ment_bound = (start_index, end_index)
    train_data = list()
    train_query_examples = io_utils.read_query_examples_file("resources/train/wec_es_train_small_qsent.json")
    print("Done loading examples file")
    train_passages = io_utils.read_passages_file("resources/train/wec_es_dev_passages.json")
    print("Done loading passages file")
    for train_exmpl in train_query_examples.values():
        # Generating positive examples
        train_data.extend(
            [
                model.convert_example_to_features(train_exmpl, train_passages[pos_id], max_query_length,
                                                  max_passage_length, True)
                for pos_id in train_exmpl["positivePassagesIds"]
            ])

        # Generating negative examples
        train_data.extend(
            [
                model.convert_example_to_features(train_exmpl, train_passages[neg_id], max_query_length,
                                                  max_passage_length, False)
                for neg_id in train_exmpl["negativePassagesIds"]
            ])

    print("Done generating training data, total of-" + str(len(train_data)) + " train examples")
    return train_data


def generate_data_loader_batches(train_features: List[InputFeature], batch_size: int):
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_start_positions, all_end_positions)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    train_batches = [batch for batch in train_dataloader]
    return train_batches


if __name__ == '__main__':
    train()
