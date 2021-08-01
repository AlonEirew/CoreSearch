from typing import List

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.data_obj import InputFeature


def generate_train_batches(train_features: List[InputFeature], batch_size: int):
    all_passage_input_ids, all_query_input_ids, \
    all_passage_input_mask, all_query_input_mask, \
    all_passage_segment_ids, all_query_segment_ids, \
    passage_event_starts, passage_event_ends = batches_essentials(train_features)
    train_data = TensorDataset(all_passage_input_ids, all_query_input_ids,
                               all_passage_input_mask, all_query_input_mask,
                               all_passage_segment_ids, all_query_segment_ids,
                               passage_event_starts, passage_event_ends)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    train_batches = [batch for batch in train_dataloader]
    return train_batches


def generate_dev_batches(train_features: List[InputFeature], batch_size: int):
    all_passage_input_ids, all_query_input_ids, \
    all_passage_input_mask, all_query_input_mask, \
    all_passage_segment_ids, all_query_segment_ids, \
    passage_event_starts, passage_event_ends = batches_essentials(train_features)

    query_start_positions = torch.tensor([f.query_event_start for f in train_features], dtype=torch.long)
    query_end_positions = torch.tensor([f.query_event_end for f in train_features], dtype=torch.long)
    passage_end_bound = torch.tensor([f.passage_end_bound for f in train_features], dtype=torch.long)
    is_positives = torch.tensor([f.is_positive for f in train_features], dtype=torch.bool)
    train_data = TensorDataset(all_passage_input_ids, all_query_input_ids,
                               all_passage_input_mask, all_query_input_mask,
                               all_passage_segment_ids, all_query_segment_ids,
                               passage_event_starts, passage_event_ends, query_start_positions, query_end_positions,
                               passage_end_bound, is_positives)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    train_batches = [batch for batch in train_dataloader]
    return train_batches


def batches_essentials(train_features: List[InputFeature]):
    all_passage_input_ids = torch.tensor([f.passage_input_ids for f in train_features], dtype=torch.long)
    all_query_input_ids = torch.tensor([f.query_input_ids for f in train_features], dtype=torch.long)

    all_passage_input_mask = torch.tensor([f.passage_input_mask for f in train_features], dtype=torch.long)
    all_query_input_mask = torch.tensor([f.query_input_mask for f in train_features], dtype=torch.long)

    all_passage_segment_ids = torch.tensor([f.passage_segment_ids for f in train_features], dtype=torch.long)
    all_query_segment_ids = torch.tensor([f.query_segment_ids for f in train_features], dtype=torch.long)

    all_start_positions = torch.tensor([f.passage_event_start for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.passage_event_end for f in train_features], dtype=torch.long)

    return all_passage_input_ids, all_query_input_ids, \
           all_passage_input_mask, all_query_input_mask, \
           all_passage_segment_ids, all_query_segment_ids, \
           all_start_positions, all_end_positions
