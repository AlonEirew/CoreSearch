import json
import os
from typing import Dict, List, Any

import torch
from haystack import Document
from tqdm import tqdm

from src.data_obj import Query, Passage, Cluster, TrainExample


def load_json_file(json_file: str):
    assert json_file
    with open(json_file, "r") as fis:
        return json.load(fis)


def read_train_example_file(train_exp_file: str) -> List[TrainExample]:
    examples_json = load_json_file(train_exp_file)
    train_exmpl = list()
    for train_exp_obj in tqdm(examples_json, desc="Reading Train Exp"):
        train_exmpl.append(TrainExample(train_exp_obj))
    return train_exmpl


def read_query_file(queries_file: str) -> List[Query]:
    queries_json = load_json_file(queries_file)
    queries = list()
    for query_obj in tqdm(queries_json, desc="Reading queries"):
        queries.append(Query(query_obj))
    return queries


def read_passages_file(passages_file: str) -> List[Passage]:
    passages_json = load_json_file(passages_file)
    passages = list()
    for pass_obj in tqdm(passages_json, desc="Reading passages"):
        passages.append(Passage(pass_obj))
    return passages


def read_gold_file(gold_file: str) -> List[Cluster]:
    gold_json = load_json_file(gold_file)
    golds = list()
    for gold_obj in gold_json:
        golds.append(Cluster(gold_obj))

    return golds


def read_id_sent_file(in_file: str) -> Dict[str, str]:
    queries = dict()
    with open(in_file, "r") as fis:
        readlines = fis.readlines()
        for line in readlines:
            line_splt = line.strip().split("\t")
            queries[line_splt[0]] = line_splt[1]

    return queries


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


def read_wec_to_haystack_doc_list(passages_file: str) -> List[Document]:
    passage_dict = read_passages_file(passages_file)
    documents: List[Document] = []
    for passage in tqdm(passage_dict, desc="Converting passages"):
        meta: Dict[str, Any] = dict()
        meta["mention"] = " ".join(passage.mention)
        meta["startIndex"] = passage.startIndex
        meta["endIndex"] = passage.endIndex
        meta["goldChain"] = passage.goldChain
        documents.append(
            Document(
                text=" ".join(passage.context),
                id=passage.id,
                meta=meta
            )
        )

    return documents
