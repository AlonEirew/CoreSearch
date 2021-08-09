import random
from typing import List

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.data_obj import SearchFeat


def generate_batches(search_features: List[SearchFeat], batch_size: int, neg_sample_size: int):
    all_passage_input_ids, all_query_input_ids, \
           all_passage_input_mask, all_query_input_mask, \
           all_passage_segment_ids, all_query_segment_ids, \
           passage_event_starts, passage_event_ends, all_end_bounds, \
           all_query_starts, all_query_ends, positive_idxs = generate_span_feats(search_features, neg_sample_size)

    assert len(all_passage_input_mask) / (neg_sample_size + 1) == len(positive_idxs)

    data = TensorDataset(all_passage_input_ids, all_query_input_ids,
                               all_passage_input_mask, all_query_input_mask,
                               all_passage_segment_ids, all_query_segment_ids,
                               passage_event_starts, passage_event_ends,
                               all_end_bounds, all_query_starts, all_query_ends)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    batches = [batch for batch in dataloader]
    pos_batch_size = batch_size / (neg_sample_size + 1)
    assert pos_batch_size.is_integer()
    pos_batch_size = int(pos_batch_size)
    positive_batches = torch.tensor([positive_idxs[x:x+pos_batch_size] for x in range(0, len(positive_idxs), pos_batch_size)])
    return batches, positive_batches


def generate_span_feats(search_features: List[SearchFeat], neg_sample_size):
    passages_input_ids = list()
    query_input_ids = list()
    passage_input_mask = list()
    query_input_mask = list()
    passage_segment_ids = list()
    query_segment_ids = list()
    passage_start_position = list()
    passage_end_position = list()
    passage_end_bound = list()
    query_event_start = list()
    query_event_end = list()

    positive_idxs = list()
    for search_feat in search_features:
        query_feat = search_feat.query
        pos_passage_feats = search_feat.positive_passage
        neg_passage_feat = search_feat.negative_passages

        for neg_feat in neg_passage_feat:
            passages_input_ids.append(neg_feat.passage_input_ids)
            query_input_ids.append(query_feat.query_input_ids)
            passage_input_mask.append(neg_feat.passage_input_mask)
            query_input_mask.append(query_feat.query_input_mask)
            passage_segment_ids.append(neg_feat.passage_segment_ids)
            query_segment_ids.append(query_feat.query_segment_ids)
            passage_start_position.append(neg_feat.passage_event_start)
            passage_end_position.append(neg_feat.passage_event_end)
            passage_end_bound.append(neg_feat.passage_end_bound)
            query_event_start.append(query_feat.query_event_start)
            query_event_end.append(query_feat.query_event_end)

        pos_idx = random.randint(len(query_event_start) - neg_sample_size, len(query_event_start))
        positive_idxs.append(pos_idx % (neg_sample_size + 1))

        passages_input_ids.insert(pos_idx, pos_passage_feats.passage_input_ids)
        query_input_ids.insert(pos_idx, query_feat.query_input_ids)
        passage_input_mask.insert(pos_idx, pos_passage_feats.passage_input_mask)
        query_input_mask.insert(pos_idx, query_feat.query_input_mask)
        passage_segment_ids.insert(pos_idx, pos_passage_feats.passage_segment_ids)
        query_segment_ids.insert(pos_idx, query_feat.query_segment_ids)
        passage_start_position.insert(pos_idx, pos_passage_feats.passage_event_start)
        passage_end_position.insert(pos_idx, pos_passage_feats.passage_event_end)
        passage_end_bound.insert(pos_idx, pos_passage_feats.passage_end_bound)
        query_event_start.insert(pos_idx, query_feat.query_event_start)
        query_event_end.insert(pos_idx, query_feat.query_event_end)

    all_passage_input_ids = torch.tensor([f for f in passages_input_ids], dtype=torch.long)
    all_query_input_ids = torch.tensor([f for f in query_input_ids], dtype=torch.long)

    all_passage_input_mask = torch.tensor([f for f in passage_input_mask], dtype=torch.long)
    all_query_input_mask = torch.tensor([f for f in query_input_mask], dtype=torch.long)

    all_passage_segment_ids = torch.tensor([f for f in passage_segment_ids], dtype=torch.long)
    all_query_segment_ids = torch.tensor([f for f in query_segment_ids], dtype=torch.long)

    all_start_positions = torch.tensor([f for f in passage_start_position], dtype=torch.long)
    all_end_positions = torch.tensor([f for f in passage_end_position], dtype=torch.long)
    all_end_bounds = torch.tensor([f for f in passage_end_bound], dtype=torch.long)
    all_query_starts = torch.tensor([f for f in query_event_start], dtype=torch.long)
    all_query_ends = torch.tensor([f for f in query_event_end], dtype=torch.long)

    return all_passage_input_ids, all_query_input_ids, \
           all_passage_input_mask, all_query_input_mask, \
           all_passage_segment_ids, all_query_segment_ids, \
           all_start_positions, all_end_positions, all_end_bounds, \
           all_query_starts, all_query_ends, positive_idxs
