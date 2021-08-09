import random

import numpy as np
import torch

from src.model import SpanPredAuxiliary
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
    max_passage_length = 150
    remove_qbound_tokens = False
    checkpoint_path = "/home/alon_nlp/event-search/checkpoints/01082021_111218"
    model_file = checkpoint_path + "/model-4.pt"
    tokenizer_path = checkpoint_path + "/tokenizer"

    dev_examples_file = "resources/train/wec_es_Dev_qsent_psegment_examples.json"
    dev_passages_file = "resources/train/wec_es_Dev_passages_segment.json"

    tokenization = Tokenization(tokenizer_path)
    model = SpanPredAuxiliary(len(tokenization.tokenizer))
    model.to(device)
    load_checkpoint(model_file, model)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    dev_data = tokenization.read_and_gen_features(dev_examples_file, dev_passages_file, max_query_length,
                                                  max_passage_length, remove_qbound_tokens)
    dev_batches = generate_dev_batches(dev_data, batch_size)

    evaluate(model, dev_batches, device)


if __name__ == '__main__':
    inference()
