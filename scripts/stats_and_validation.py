from haystack.modeling.data_handler.processor import _read_squad_file

from src.override_classes.retriever.wec_processor import WECSimilarityProcessor

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
    read_query_files("data/resources/WEC-ES/clean/Train_queries.json",
                     "data/resources/WEC-ES/clean/Dev_queries.json",
                     "data/resources/WEC-ES/clean/Test_queries.json")
    print("########")

    print("#### Total Passage collections ####")
    read_passage_files("data/resources/WEC-ES/clean/Train_all_passages.json",
                       "data/resources/WEC-ES/clean/Dev_all_passages.json",
                       "data/resources/WEC-ES/clean/Test_all_passages.json")
    print("########")

    print("#### Total Clusters ####")
    train_clusters = io_utils.read_gold_file("data/resources/WEC-ES/clean/Train_gold_clusters.json")
    dev_clusters = io_utils.read_gold_file("data/resources/WEC-ES/clean/Dev_gold_clusters.json")
    test_clusters = io_utils.read_gold_file("data/resources/WEC-ES/clean/Test_gold_clusters.json")
    print("Train Clusters=" + str(len(train_clusters)))
    print("Dev Clusters=" + str(len(dev_clusters)))
    print("Test Clusters=" + str(len(test_clusters)))
    print("########")

    print("#### Total Mentions (should equal the number of queries) ####")
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
    read_query_files("data/resources/WEC-ES/train/Train_queries.json",
                     "data/resources/WEC-ES/train/Dev_queries.json",
                     "data/resources/WEC-ES/train/Test_queries.json")
    print("########")

    print("#### Total Passages for Training (that serve as positive or negative examples) ####")
    read_passage_files("data/resources/WEC-ES/train/Train_passages.json",
                       "data/resources/WEC-ES/train/Dev_passages.json",
                       "data/resources/WEC-ES/train/Test_passages.json")
    print("########")


def run_dpr_stats(train_clusters, dev_clusters, test_clusters):
    print("#### Total queries for retriever (DPR format) ==sum(each cluster mentions)^2)-size(cluster) ####")
    train_dpr = WECSimilarityProcessor._read_dpr_json("data/resources/dpr/context/Train.json")
    dev_dpr = WECSimilarityProcessor._read_dpr_json("data/resources/dpr/context/Dev.json")
    test_dpr = WECSimilarityProcessor._read_dpr_json("data/resources/dpr/context/Test.json")
    print(f"Train Queries={len(train_dpr)}=={get_sum_dpr_queries_from_clusters(train_clusters)}")
    print(f"Dev Queries={len(dev_dpr)}=={get_sum_dpr_queries_from_clusters(dev_clusters)}")
    print(f"Test Queries={len(test_dpr)}=={get_sum_dpr_queries_from_clusters(test_clusters)}")
    print("########")


def run_squad_stats():
    print("#### Total queries for reader (SQUAD format)")
    train_squad = _read_squad_file("data/resources/squad/context/Train_squad_format_1pos_24neg.json")
    dev_squad = _read_squad_file("data/resources/squad/context/Dev_squad_format_1pos_24neg.json")
    test_squad = _read_squad_file("data/resources/squad/context/Test_squad_format_1pos_24neg.json")
    generate_squad_stats(train_squad, "Train")
    generate_squad_stats(dev_squad, "dev")
    generate_squad_stats(test_squad, "Test")
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


if __name__ == '__main__':
    train_clusters, dev_clusters, test_clusters = run_clean_stats()
    run_train_stats()
    run_dpr_stats(train_clusters, dev_clusters, test_clusters)
    run_squad_stats()
    print("Done!")
