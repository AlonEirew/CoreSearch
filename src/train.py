import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import AdamW

from src import io_utils
from src.data.input_feature import InputFeature
from src.model import WecEsModel


def train():
    train_examples_file = "resources/train/wec_es_train_small_qsent.json"
    train_passages_file = "resources/train/wec_es_train_passages.json"
    dev_examples_file = "resources/train/wec_es_train_small_qsent.json"
    dev_passages_file = "resources/train/wec_es_train_passages.json"

    cpu_only = False
    epochs = 100
    batch_size = 32
    lr = 1e-5
    # hidden_size = 500
    max_query_length = 50
    max_passage_length = 200
    assert (max_query_length + max_passage_length + 3) <= 512
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu_only else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1243)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(1234)

    model = WecEsModel()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    train_data = read_and_gen_features(model, train_examples_file, train_passages_file, max_query_length, max_passage_length)
    train_batches = generate_train_batches(train_data, batch_size)
    dev_data = read_and_gen_features(model, dev_examples_file, dev_passages_file, max_query_length, max_passage_length)
    dev_batches = generate_dev_batches(dev_data, batch_size)

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
            input_ids, input_mask, segment_ids, start_position, end_position = batch
            outputs = model(input_ids, input_mask, segment_ids, start_position, end_position)
            loss = outputs.loss
            accum_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tot_steps += 1
            print('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                epoch, step + 1, len(train_batches), time.time() - start_time, accum_loss / tot_steps))

        evaluate(model, dev_batches, device)


def evaluate(model, dev_batches, device):
    model.eval()
    start_lab, start_pred, end_lab, end_pred = (list(), list(), list(), list())
    for step, batch in enumerate(dev_batches):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, pass_start, pass_end, query_start, query_end, is_positive = batch
        with torch.no_grad():
            outputs = model(input_ids, input_mask, segment_ids, pass_start, pass_end)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        for res_ind in range(start_logits.shape[0]):
            start_tolist = start_logits[res_ind].detach().cpu().tolist()
            end_tolist = end_logits[res_ind].detach().cpu().tolist()
            sorted_start_logits = sorted(enumerate(start_tolist), key=lambda x: x[1], reverse=True)
            sorted_end_logits = sorted(enumerate(end_tolist), key=lambda x: x[1], reverse=True)

            start_lab.append(pass_start.data[res_ind].item())
            start_pred.append(sorted_start_logits[0][0])
            end_lab.append(pass_end.data[res_ind].item())
            end_pred.append(sorted_end_logits[0][0])

    s_precision, s_recall, s_f1, _ = precision_recall_fscore_support(start_lab, start_pred, average='macro', zero_division=0)
    e_precision, e_recall, e_f1, _ = precision_recall_fscore_support(end_lab, end_pred, average='macro', zero_division=0)
    s_accuracy = accuracy_score(start_lab, start_pred)
    e_accuracy = accuracy_score(end_lab, end_pred)
    print("Start Position: accuracy={}, precision={}, recall={}, f1={}".format(s_accuracy, s_precision, s_recall, s_f1))
    print("End Position: accuracy={}, precision={}, recall={}, f1={}".format(e_accuracy, e_precision, e_recall, e_f1))
    # print("Avg Position: precision={}, recall={}, f1={}".format(e_precision, e_recall, e_f1))


def read_and_gen_features(model, exmpl_file, passage_file, max_query_length, max_passage_length) -> List[InputFeature]:
    query_examples = io_utils.read_query_examples_file(exmpl_file)
    print("Done loading examples file-" + exmpl_file)
    passages = io_utils.read_passages_file(passage_file)
    print("Done loading passages file-" + passage_file)

    # list of (query_id, passage_id, pass_ment_bound, gold) where pass_ment_bound = (start_index, end_index)
    data = list()
    print("Starting to generate examples...")
    for exmpl in tqdm(query_examples.values()):
        # Generating positive examples
        data.extend(
            [
                model.convert_example_to_features(exmpl, passages[pos_id], max_query_length, max_passage_length, True)
                for pos_id in exmpl["positivePassagesIds"]
            ])

        # Generating negative examples
        data.extend(
            [
                model.convert_example_to_features(exmpl, passages[neg_id], max_query_length, max_passage_length, False)
                for neg_id in exmpl["negativePassagesIds"]
            ])

    print("Done generating data, total of-" + str(len(data)) + " train examples")
    return data


def generate_train_batches(train_features: List[InputFeature], batch_size: int):
    input_ids, input_masks, segment_ids, start_positions, end_positions = batches_essentials(train_features)
    train_data = TensorDataset(input_ids, input_masks, segment_ids,
                               start_positions, end_positions)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    train_batches = [batch for batch in train_dataloader]
    return train_batches


def generate_dev_batches(train_features: List[InputFeature], batch_size: int):
    input_ids, input_masks, segment_ids, pass_start_positions, pass_end_positions = batches_essentials(train_features)
    query_start_positions = torch.tensor([f.query_start_position for f in train_features], dtype=torch.long)
    query_end_positions = torch.tensor([f.query_end_position for f in train_features], dtype=torch.long)
    is_positives = torch.tensor([f.is_positive for f in train_features], dtype=torch.bool)
    train_data = TensorDataset(input_ids, input_masks, segment_ids,
                               pass_start_positions, pass_end_positions, query_start_positions, query_end_positions,
                               is_positives)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    train_batches = [batch for batch in train_dataloader]
    return train_batches


def batches_essentials(train_features: List[InputFeature]):
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.passage_start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.passage_end_position for f in train_features], dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions


if __name__ == '__main__':
    train()
