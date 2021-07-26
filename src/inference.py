import random

import numpy as np
import torch

from src.model import WecEsModel
from src.utils.dataset_utils import generate_dev_batches
from src.utils.evaluation import evaluate
from src.utils.io_utils import load_checkpoint
from src.utils.tokenization import Tokenization


def inference():
    cpu_only = False
    n_gpu = torch.cuda.device_count()
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1243)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(1234)

    device = torch.device("cuda" if torch.cuda.is_available() and not cpu_only else "cpu")
    batch_size = 32
    max_query_length = 50
    max_passage_length = 200
    checkpoint_path = "/home/alon_nlp/event-search/checkpoints/21072021_110812"
    model_file = checkpoint_path + "/model-9.pt"
    tokenizer_path = checkpoint_path + "/tokenizer"

    dev_examples_file = "resources/train/wec_es_dev_qsent_examples.json"
    dev_passages_file = "resources/train/wec_es_dev_passages.json"

    tokenization = Tokenization(tokenizer_path)
    model = WecEsModel(len(tokenization.tokenizer))
    model.to(device)
    load_checkpoint(model_file, model)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    dev_data = tokenization.read_and_gen_features(dev_examples_file, dev_passages_file, max_query_length, max_passage_length)
    dev_batches = generate_dev_batches(dev_data, batch_size)

    evaluate(model, dev_batches, device)


if __name__ == '__main__':
    inference()
