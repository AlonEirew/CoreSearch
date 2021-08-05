from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from src.data_obj import SearchFeat, QueryFeat, PassageFeat


class SearchFeatDataset(Dataset):
    def __init__(self, search_feats: List[SearchFeat]):
        self.search_features = search_feats

    def __len__(self):
        return len(self.search_features)

    def __getitem__(self, idx):
        result = self.search_features[idx]
        return result.query_id, result.positive_passage_id, result.negative_passages_ids

    @staticmethod
    def generate_search_batches(search_features: List[SearchFeat], batch_size: int):
        train_data = SearchFeatDataset(search_features)
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        train_batches = [batch for batch in train_dataloader]
        return train_batches

    @staticmethod
    def generate_span_feats(query_feats: Dict[int, QueryFeat], passages_feats: Dict[int, PassageFeat],
                            query_id: int, positive_id: int, negative_ids: List[int], device):
        query_feat = query_feats[query_id]
        pos_passage_feats = passages_feats[positive_id]
        neg_passage_feat = [passages_feats[neg_id] for neg_id in negative_ids]

        passages_input_ids = [pos_passage_feats.passage_input_ids]
        query_input_ids = [query_feat.query_input_ids]
        passage_input_mask = [pos_passage_feats.passage_input_mask]
        query_input_mask = [query_feat.query_input_mask]
        passage_segment_ids = [pos_passage_feats.passage_segment_ids]
        query_segment_ids = [query_feat.query_segment_ids]
        passage_start_position = [pos_passage_feats.passage_event_start]
        passage_end_position = [pos_passage_feats.passage_event_end]
        for neg_feat in neg_passage_feat:
            passages_input_ids.append(neg_feat.passage_input_ids)
            query_input_ids.append(query_feat.query_input_ids)
            passage_input_mask.append(neg_feat.passage_input_mask)
            query_input_mask.append(query_feat.query_input_mask)
            passage_segment_ids.append(neg_feat.passage_segment_ids)
            query_segment_ids.append(query_feat.query_segment_ids)
            passage_start_position.append(neg_feat.passage_event_start)
            passage_end_position.append(neg_feat.passage_event_end)

        all_passage_input_ids = torch.tensor([f for f in passages_input_ids], dtype=torch.long, device=device)
        all_query_input_ids = torch.tensor([f for f in query_input_ids], dtype=torch.long, device=device)

        all_passage_input_mask = torch.tensor([f for f in passage_input_mask], dtype=torch.long, device=device)
        all_query_input_mask = torch.tensor([f for f in query_input_mask], dtype=torch.long, device=device)

        all_passage_segment_ids = torch.tensor([f for f in passage_segment_ids], dtype=torch.long, device=device)
        all_query_segment_ids = torch.tensor([f for f in query_segment_ids], dtype=torch.long, device=device)

        all_start_positions = torch.tensor([f for f in passage_start_position], dtype=torch.long, device=device)
        all_end_positions = torch.tensor([f for f in passage_end_position], dtype=torch.long, device=device)

        return all_passage_input_ids, all_query_input_ids, \
               all_passage_input_mask, all_query_input_mask, \
               all_passage_segment_ids, all_query_segment_ids, \
               all_start_positions, all_end_positions
