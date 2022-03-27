from typing import List

from haystack.modeling.data_handler.samples import SampleBasket, Sample

from src.override_classes.wec_context_processor import WECContextProcessor


class WECStartEndProcessor(WECContextProcessor):
    def __init__(
        self,
        query_tokenizer,
        passage_tokenizer,
        max_seq_len_query,
        max_seq_len_passage,
        data_dir="",
        metric=None,
        train_filename="train.json",
        dev_filename=None,
        test_filename="test.json",
        dev_split=0.1,
        proxies=None,
        max_samples=None,
        embed_title=True,
        num_positives=1,
        num_hard_negatives=1,
        shuffle_negatives=True,
        shuffle_positives=False,
        label_list=None,
        add_spatial_tokens=None
    ):
        super(WECStartEndProcessor, self).__init__(
            query_tokenizer,
            passage_tokenizer,
            max_seq_len_query,
            max_seq_len_passage,
            data_dir,
            metric,
            train_filename,
            dev_filename,
            test_filename,
            dev_split,
            proxies,
            max_samples,
            embed_title,
            num_positives,
            num_hard_negatives,
            shuffle_negatives,
            shuffle_positives,
            label_list,
            add_spatial_tokens
        )

    def _convert_queries(self, baskets: List[SampleBasket]):
        assert "query_id" in baskets[0].raw and "start_index" in baskets[0].raw and "end_index" in baskets[0].raw, "Invalid sample"
        for basket in baskets:
            # extract query, positive context passages and titles, hard-negative passages and titles
            clear_text, tokenized, features, query_feat = self.get_basket_tokens_feats(basket)
            if query_feat is not None and len(features) > 0:
                features[0]["query_start"] = query_feat.query_event_start
                features[0]["query_end"] = query_feat.query_event_end
            sample = Sample(id="",
                            clear_text=clear_text,
                            tokenized=tokenized,
                            features=features)  # type: ignore
            basket.samples = [sample]
        return baskets
