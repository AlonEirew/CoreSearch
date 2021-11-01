from typing import List, Dict

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.data_obj import SearchFeat, QueryResult, Cluster


def generate_batches(search_features: List[SearchFeat], batch_size: int):
    all_passage_input_ids, all_query_input_ids, \
    all_passage_input_mask, all_query_input_mask, \
    all_passage_segment_ids, all_query_segment_ids, \
    passage_event_starts, passage_event_ends, all_end_bounds, \
    all_query_starts, all_query_ends = generate_span_feats(search_features)

    data = TensorDataset(all_passage_input_ids, all_query_input_ids,
                         all_passage_input_mask, all_query_input_mask,
                         all_passage_segment_ids, all_query_segment_ids,
                         passage_event_starts, passage_event_ends,
                         all_end_bounds, all_query_starts, all_query_ends)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    batches = [batch for batch in dataloader]
    return batches


def generate_span_feats(search_features: List[SearchFeat]):
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

    # positive_idxs = list()
    for search_feat in search_features:
        query_feat = search_feat.query
        pos_passage_feats = search_feat.positive_passage
        neg_passage_feat = search_feat.negative_passages

        passages_input_ids.append(pos_passage_feats.passage_input_ids)
        query_input_ids.append(query_feat.query_input_ids)
        passage_input_mask.append(pos_passage_feats.passage_input_mask)
        query_input_mask.append(query_feat.query_input_mask)
        passage_segment_ids.append(pos_passage_feats.passage_segment_ids)
        query_segment_ids.append(query_feat.query_segment_ids)
        passage_start_position.append(pos_passage_feats.passage_event_start)
        passage_end_position.append(pos_passage_feats.passage_event_end)
        passage_end_bound.append(pos_passage_feats.passage_end_bound)
        query_event_start.append(query_feat.query_event_start)
        query_event_end.append(query_feat.query_event_end)

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
           all_query_starts, all_query_ends


def query_results_to_ids_list(query_results: List[QueryResult]) -> Dict[str, List[str]]:
    assert query_results
    new_queries = dict()
    for query_res in query_results:
        new_queries[query_res.query.id] = list()
        for passage in query_res.results:
            new_queries[query_res.query.id].append(passage.id)
    return new_queries


def clusters_to_ids_list(gold_clusters: List[Cluster]) -> Dict[str, List[str]]:
    assert gold_clusters
    new_results = dict()
    for clust in gold_clusters:
        for id_ in clust.mention_ids:
            new_ment_list = clust.mention_ids.copy()
            new_ment_list.remove(id_)
            new_results[id_] = new_ment_list
    return new_results
