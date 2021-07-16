import random

import numpy as np
import torch

from src import utils


def train():
    train_queries, train_passages, train_gold = read_resources()
    n_gpu = torch.cuda.device_count()
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1243)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(1234)


def read_resources():
    train_queries = utils.read_queries_file("resources/queries/Train_event_sent_queries.tsv")
    train_passages = utils.read_queries_file("resources/passages/Train_passages.tsv")
    train_gold = utils.read_gold_file("resources/gold/Train_query_to_relevant_passages.tsv")

    return train_queries, train_passages, train_gold


if __name__ == '__main__':
    pass
