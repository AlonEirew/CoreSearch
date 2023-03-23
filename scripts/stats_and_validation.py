import random

from haystack.modeling.data_handler.processor import _read_squad_file
from src.override_classes.retriever.search_processor import CoreSearchSimilarityProcessor

from src.utils import io_utils


def read_query_files(train_file, dev_file, test_file):
    train_queries = io_utils.read_query_file(train_file)
    dev_queries = io_utils.read_query_file(dev_file)
    test_queries = io_utils.read_query_file(test_file)
    print("Train Queries=" + str(len(train_queries)))
    print("Dev Queries=" + str(len(dev_queries)))
    print("Test Queries=" + str(len(test_queries)))


def read_passage_files(train_file, dev_file, test_file):
    train_passages = io_utils.read_passages_file(train_file)
    dev_passages = io_utils.read_passages_file(dev_file)
    test_passages = io_utils.read_passages_file(test_file)
    print("Train Passages=" + str(len(train_passages)))
    print("Dev Passages=" + str(len(dev_passages)))
    print("Test Passages=" + str(len(test_passages)))


def get_cluster_mentions(clusters):
    total_mentions = 0
    for clust in clusters:
        total_mentions += len(clust.mention_ids)
    return total_mentions


def get_sum_dpr_queries_from_clusters(clusters):
    total_mentions = 0
    for clust in clusters:
        total_mentions += (pow(len(clust.mention_ids), 2) - len(clust.mention_ids))
    return total_mentions


def run_clean_stats():
    print("#### Total Queries ####")
    read_query_files("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/min_span/Train_queries_min_span.json",
                     "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/min_span/Dev_queries_min_span.json",
                     "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/min_span/Test_queries_min_span.json")
    print("########")

    print("#### Total Passage collections ####")
    read_passage_files("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/min_span/Train_passages_min_span.json",
                       "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/min_span/Dev_passages_min_span.json",
                       "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/min_span/Test_passages_min_span.json")
    print("########")

    print("#### Total Clusters ####")
    train_clusters = io_utils.read_gold_file("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/min_span/Train_gold_clusters_min_span.json")
    dev_clusters = io_utils.read_gold_file("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/min_span/Dev_gold_clusters_min_span.json")
    test_clusters = io_utils.read_gold_file("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/clean/min_span/Test_gold_clusters_min_span.json")
    print("Train Clusters=" + str(len(train_clusters)))
    print("Dev Clusters=" + str(len(dev_clusters)))
    print("Test Clusters=" + str(len(test_clusters)))
    print("########")

    print("#### Total Mentions (generated from the clusters -- should equal the number of queries) ####")
    train_ments = get_cluster_mentions(train_clusters)
    dev_ments = get_cluster_mentions(dev_clusters)
    test_ments = get_cluster_mentions(test_clusters)
    print("Train Mentions=" + str(train_ments))
    print("Dev Mentions=" + str(dev_ments))
    print("Test Mentions=" + str(test_ments))
    print("########")

    return train_clusters, dev_clusters, test_clusters


def run_train_stats():
    print("#### Total Queries for training (should be the same as total queries) ####")
    read_query_files("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/train/min_span/Train_queries_min_span.json",
                     "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/train/min_span/Dev_queries_min_span.json",
                     "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/train/min_span/Test_queries_min_span.json")
    print("########")

    print("#### Total Passages for Training (that serve as positive or negative examples) ####")
    read_passage_files("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/train/min_span/Train_passages_min_span.json",
                       "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/train/min_span/Dev_passages_min_span.json",
                       "/Users/aloneirew/workspace/Datasets/CoreSearchDataset/train/min_span/Test_passages_min_span.json")
    print("########")


def run_dpr_stats(train_clusters, dev_clusters, test_clusters):
    print("#### Total queries for retriever (DPR format) ==sum(each cluster mentions)^2)-size(cluster) ####")
    train_dpr = CoreSearchSimilarityProcessor._read_dpr_json("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/dpr/min_span/Train.json")
    dev_dpr = CoreSearchSimilarityProcessor._read_dpr_json("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/dpr/min_span/Dev.json")
    test_dpr = CoreSearchSimilarityProcessor._read_dpr_json("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/dpr/min_span/Test.json")
    print(f"Train Queries={len(train_dpr)}=={get_sum_dpr_queries_from_clusters(train_clusters)}")
    print(f"Dev Queries={len(dev_dpr)}=={get_sum_dpr_queries_from_clusters(dev_clusters)}")
    print(f"Test Queries={len(test_dpr)}=={get_sum_dpr_queries_from_clusters(test_clusters)}")
    print("########")


def run_squad_stats():
    print("#### Total queries for reader (SQUAD format)")
    train_squad = _read_squad_file("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/squad/min_span/Train_squad_format_1pos_23neg.json")
    dev_squad = _read_squad_file("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/squad/min_span/Dev_squad_format_1pos_23neg.json")
    test_squad = _read_squad_file("/Users/aloneirew/workspace/Datasets/CoreSearchDataset/squad/min_span/Test_squad_format_1pos_23neg.json")
    generate_squad_stats(train_squad, "Train")
    generate_squad_stats(dev_squad, "Tev")
    generate_squad_stats(test_squad, "Test")

    print("### Validating Squad Format")
    run_squad_validation(train_squad)
    run_squad_validation(dev_squad)
    run_squad_validation(test_squad)
    print("########")


def generate_squad_stats(squad_set_data, set_str):
    num_contexts = 0
    total_queries = 0
    pos_queries = 0
    neg_queries = 0
    for paragraphs_dict in squad_set_data:
        paragraphs = paragraphs_dict['paragraphs']
        num_contexts += len(paragraphs)
        for qas_dict in paragraphs:
            qas = qas_dict['qas']
            total_queries += len(qas)
            for query in qas:
                is_impossible = query['is_impossible']
                if is_impossible:
                    neg_queries += 1
                else:
                    pos_queries += 1

    print(f"{set_str} Queries={str(total_queries)}")
    print(f"{set_str} Contexts={str(num_contexts)}")
    print(f"{set_str} Positive Queries={str(pos_queries)}")
    print(f"{set_str} Negative Queries={str(neg_queries)}")
    print("########")


def run_squad_validation(set_to_validate):
    for paragraphs_dict in set_to_validate:
        paragraphs = paragraphs_dict['paragraphs']
        for qas_dict in paragraphs:
            context = qas_dict['context'].split(" ")
            qas = qas_dict['qas']
            for query in qas:
                question = query['question'].split(" ")
                answers = query['answers']
                ment_start = query['ment_start']
                ment_end = query['ment_end']
                query_mention_qst = question[ment_start:ment_end + 1]
                query_mention = query['query_mention']
                if query_mention_qst != query_mention:
                    print(f"Mention_in_query={query_mention_qst} not aligned with mention text={query_mention}, for mentionID=" + query['id'])
                if not query['is_impossible']:
                    for answer in answers:
                        answer_text = answer['text'].split(" ")
                        answer_start = answer['answer_start']
                        answer_ment_start = answer['ment_start']
                        answer_ment_end = answer['ment_end']
                        answer_mention = context[answer_ment_start:answer_ment_end + 1]
                        context_joind = " ".join(context)
                        answer_from_joined = context_joind[answer_start:answer_start + len(answer['text'])]
                        if answer_mention != answer_text:
                            print(f"Mention_in_answer={answer_mention} not aligned with mention text={answer_text}, for mentionID=" + query['id'])
                        if answer_from_joined != answer['text']:
                            print(f"Mention_in_answer={answer_mention} not aligned with mention text={answer_text}, for mentionID=" + query['id'])

    print("# Done Validation! If empty passed #")


# def arrange_by_queries(set_to_validate):
#     pre_buckets = list()
#     for paragraphs_dict in set_to_validate:
#         paragraphs = paragraphs_dict['paragraphs']
#         for qas_dict in paragraphs:
#             qas = qas_dict['qas']
#             pre_buckets.extend(qas)
#
#     query_pos_to_passage = dict()
#     query_neg_to_passage = dict()
#     set_query_ids = set()
#     random.shuffle(set_to_validate)
#     for sample_bask in pre_buckets:
#         query_id = sample_bask['id']
#         set_query_ids.add(query_id)
#         if query_id not in query_pos_to_passage:
#             query_pos_to_passage[query_id] = list()
#
#         if query_id not in query_neg_to_passage:
#             query_neg_to_passage[query_id] = list()
#
#         for sample in sample_bask.samples:
#             if sample.features[0]['query_coref_link'] == sample.features[0]['passage_coref_link']:
#                 query_pos_to_passage[query_id].append(sample)
#             else:
#                 query_neg_to_passage[query_id].append(sample)
#
#     ret_baskets = list()
#     for query_id in set_query_ids:
#         query_bask = list()
#         positive_len = len(query_pos_to_passage[query_id])
#         total_in_batch = self.num_positives + self.num_negatives
#         assert positive_len >= self.num_positives
#         query_bask.extend(query_pos_to_passage[query_id][:self.num_positives])
#         query_bask.extend(query_neg_to_passage[query_id][:total_in_batch - self.num_positives])
#         assert len(query_bask) % self.batch_size == 0
#         ret_baskets.extend(query_bask)
#
#     return ret_baskets


if __name__ == '__main__':
    train_clusters, dev_clusters, test_clusters = run_clean_stats()
    run_train_stats()
    run_dpr_stats(train_clusters, dev_clusters, test_clusters)
    run_squad_stats()
    print("Done!")
