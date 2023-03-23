import json
from typing import List

from src.data_obj import Cluster
from src.utils import io_utils


def convert_to_dict(json_data_dev: List[object], json_data_test: List[object]):
    json_map = dict()
    for data in json_data_dev:
        json_map[str(data['mention_id'])] = data

    for data in json_data_test:
        json_map[str(data['mention_id'])] = data

    return json_map


def load_file_from_json(file_to_load):
    # load file from json
    with open(file_to_load) as f:
        loaded_file = json.load(f)

    return loaded_file


def write_json(output_file, data):
    with open(output_file, 'w') as f:
        json.dump(data, f, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)


def main():
    SPLIT = "Train"
    # clean or train
    SET = "clean"
    fixed_span_dev_file = "/Users/aloneirew/workspace/Datasets/WECEng/dev_fixed/Dev_Event_gold_mentions_validated_fixed_final.json"
    fixed_span_test_file = "/Users/aloneirew/workspace/Datasets/WECEng/test_fixed/Test_Event_gold_mentions_validated_fixed_final.json"
    gold_clusters = "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/min_span/" + SPLIT + "_gold_clusters_min_span.json"

    train_query_file = "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/" + SET + "/" + SPLIT + "_queries.json"
    train_passage_file = "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/" + SET + "/" + SPLIT + "_passages.json"

    fixed_span_dev_json = load_file_from_json(fixed_span_dev_file)
    fixed_span_test_json = load_file_from_json(fixed_span_test_file)
    golds: List[Cluster] = io_utils.read_gold_file(gold_clusters)

    gold_map = {int(clust.cluster_id): clust.mention_ids for clust in golds}

    train_query_json = load_file_from_json(train_query_file)
    train_passage_json = load_file_from_json(train_passage_file)

    output_query_file = "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/" + SET + "/min_span/" + SPLIT + "_queries_min_span.json"
    output_passage_file = "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/" + SET + "/min_span/" + SPLIT + "_passages_min_span.json"

    fixed_span_dict = convert_to_dict(fixed_span_dev_json, fixed_span_test_json)

    fixed_queries = list()
    queries_removed_tot = 0
    for ment in train_query_json:
        if ment['id'] in fixed_span_dict:
            fixed_ment = fixed_span_dict[ment['id']]
            if 'remove' in fixed_ment and fixed_ment['remove']:
                queries_removed_tot += 1
                continue

            if ment['goldChain'] not in gold_map or len(gold_map[ment['goldChain']]) == 1:
                queries_removed_tot += 1
                continue

            queries_left = handle_pos_examples(fixed_span_dict, ment)

            if queries_left == 0:
                queries_removed_tot += 1
                continue

            ment['mention'] = fixed_ment['tokens_str'].split()
            ment['startIndex'] = fixed_ment['tokens_number'][0]
            ment['endIndex'] = fixed_ment['tokens_number'][-1]
            fixed_queries.append(ment)

    print(f"queries_removed_tot={queries_removed_tot}")

    fixed_passages = list()
    for ment in train_passage_json:
        if ment['id'] in fixed_span_dict:
            fixed_ment = fixed_span_dict[ment['id']]
            if 'remove' in fixed_ment and fixed_ment['remove']:
                ment['remove'] = True
                fixed_passages.append(ment)
                continue

            ment['mention'] = fixed_ment['tokens_str'].split()
            ment['startIndex'] = fixed_ment['tokens_number'][0]
            ment['endIndex'] = fixed_ment['tokens_number'][-1]

        fixed_passages.append(ment)

    write_json(output_query_file, fixed_queries)
    write_json(output_passage_file, fixed_passages)


def handle_pos_examples(fixed_span_dict, ment):
    if 'positive_examples' in ment:
        fixed_pos_examples = list()
        pos_examples = ment['positive_examples']
        for ment_id in pos_examples:
            if ment_id in fixed_span_dict:
                if 'remove' in fixed_span_dict[ment_id] and fixed_span_dict[ment_id]['remove']:
                    continue
                else:
                    fixed_pos_examples.append(ment_id)

        ment['positive_examples'] = fixed_pos_examples
        return len(fixed_pos_examples)
    else:
        return -1


if __name__ == '__main__':
    main()