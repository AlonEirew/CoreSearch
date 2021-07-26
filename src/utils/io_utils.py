import json
import os
from typing import Dict, List

import torch


def read_query_examples_file(queries_file: str) -> Dict[str, Dict]:
    assert queries_file
    with open(queries_file, "r") as fis:
        queries_json = json.load(fis)
    return {qobj["id"]: qobj for qobj in queries_json}


def read_passages_file(passages_file: str) -> Dict[str, Dict]:
    assert passages_file
    with open(passages_file, "r") as fis:
        passages_json = json.load(fis)
    return {pobj["id"]: pobj for pobj in passages_json}


def read_id_sent_file(in_file: str) -> Dict[str, str]:
    queries = dict()
    with open(in_file, "r") as fis:
        readlines = fis.readlines()
        for line in readlines:
            line_splt = line.strip().split("\t")
            queries[line_splt[0]] = line_splt[1]

    return queries


def read_gold_file(gold_file: str) -> Dict[str, List[str]]:
    gold_labels = dict()
    with open(gold_file, "r") as fis:
        readlines = fis.readlines()
        for line in readlines:
            line_splt = line.strip().split("\t")
            gold_labels[line_splt[0]] = line_splt[1:]

    return gold_labels


def save_checkpoint(path, epoch, model, optimizer):
    model_file_name = os.path.join(path, "model-{}.pt".format(epoch))
    print(f"Saving a checkpoint to {model_file_name}...")

    if hasattr(model, 'module'):
        model = model.module

    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, model_file_name)


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
