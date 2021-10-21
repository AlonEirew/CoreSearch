"""
This script generate the negative samples for training, similarly to what was done in DPR.
Prerequisite for running this script is generating the Elasticsearch index using elastic_index.py script
"""
from typing import Dict

from src.utils import io_utils


def main():
    queries_file = "/Users/aeirew/workspace/wec_eng-to-wec_es/input/WEC-ES/final_sets_250821/bm25_input/Train_queries_event_sent.tsv"
    gold_file = "/Users/aeirew/workspace/wec_eng-to-wec_es/input/WEC-ES/final_sets_250821/bm25_input/Train_gold.tsv"
    result_out_file = "output/Train_bm25_pred.tsv"
    topk = 500

    query_examples: Dict[int, Dict] = io_utils.read_query_file("resources/WEC-ES/Train_queries.json")



if __name__ == '__main__':
    main()
