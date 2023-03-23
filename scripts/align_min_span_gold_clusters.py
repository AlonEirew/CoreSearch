import json
from typing import List

from scripts.align_min_span_clean import load_file_from_json, convert_to_dict, write_json
from src.data_obj import Cluster
from src.utils import io_utils


def main():
    SPLIT = "Train"
    fixed_span_dev_file = "/Users/aloneirew/workspace/Datasets/WECEng/dev_fixed/Dev_Event_gold_mentions_validated_fixed_final.json"
    fixed_span_test_file = "/Users/aloneirew/workspace/Datasets/WECEng/test_fixed/Test_Event_gold_mentions_validated_fixed_final.json"
    gold_clusters = "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/" + SPLIT + "_gold_clusters.json"

    fixed_span_dev_json =  load_file_from_json(fixed_span_dev_file)
    fixed_span_test_json = load_file_from_json(fixed_span_test_file)
    golds: List[Cluster] = io_utils.read_gold_file(gold_clusters)

    output_gold_file = "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/min_span/" + SPLIT + "_gold_clusters_min_span.json"

    fixed_span_dict = convert_to_dict(fixed_span_dev_json, fixed_span_test_json)

    queries_removed_tot = 0
    fixed_golds = list()
    for cluster in golds:
        fixed_cluster = dict()
        fixed_cluster['clusterId'] = int(cluster.cluster_id)
        fixed_cluster['clusterTitle'] = cluster.cluster_title
        fixed_cluster['mentionIds'] = list()

        for ment_id in cluster.mention_ids:
            if ment_id in fixed_span_dict:
                fixed_ment = fixed_span_dict[ment_id]
                if 'remove' in fixed_ment and fixed_ment['remove']:
                    queries_removed_tot += 1
                    continue

            fixed_cluster['mentionIds'].append(int(ment_id))

        if len(fixed_cluster['mentionIds']) > 1:
            fixed_golds.append(fixed_cluster)

    print(f"queries_removed_tot={queries_removed_tot}")
    write_json(output_gold_file, fixed_golds)


if __name__ == '__main__':
    main()