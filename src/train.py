import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AdamW

from src.model import WecEsModel
from src.utils.dataset_utils import generate_train_batches, generate_dev_batches
from src.utils.evaluation import evaluate
from src.utils.io_utils import save_checkpoint
from src.utils.tokenization import Tokenization


def train():
    start_time = datetime.now()
    dt_string = start_time.strftime("%d%m%Y_%H%M%S")

    train_examples_file = "resources/train/wec_es_train_small_qsent.json"
    train_passages_file = "resources/train/wec_es_train_passages.json"
    dev_examples_file = "resources/train/wec_es_dev_small_qsent.json"
    dev_passages_file = "resources/train/wec_es_dev_passages.json"

    tokenizer_path = "checkpoints/" + dt_string + "/tokenizer"
    checkpoints_path = "checkpoints/" + dt_string
    Path(checkpoints_path).mkdir(parents=True)
    Path(tokenizer_path).mkdir()
    print(f"{checkpoints_path}-folder created..")

    cpu_only = False
    epochs = 15
    batch_size = 32
    lr = 1e-6
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

    tokenization = Tokenization()
    tokenization.tokenizer.save_pretrained(tokenizer_path)
    model = WecEsModel(len(tokenization.tokenizer))
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=lr)

    train_data = tokenization.read_and_gen_features(train_examples_file, train_passages_file, max_query_length, max_passage_length)
    train_batches = generate_train_batches(train_data, batch_size)
    dev_data = tokenization.read_and_gen_features(dev_examples_file, dev_passages_file, max_query_length, max_passage_length)
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
