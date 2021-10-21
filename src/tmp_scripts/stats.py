from src.utils import io_utils

if __name__ == '__main__':

    train_queries = io_utils.read_query_file("resources/WEC-ES/Train_queries.json")
    dev_queries = io_utils.read_query_file("resources/WEC-ES/Dev_queries.json")
    test_queries = io_utils.read_query_file("resources/WEC-ES/Test_queries.json")
    print("Train Queries=" + str(len(train_queries)))
    print("Dev Queries=" + str(len(dev_queries)))
    print("Test Queries=" + str(len(test_queries)))

    train_passages = io_utils.read_passages_file("resources/WEC-ES/Train_passages.json")
    dev_passages = io_utils.read_passages_file("resources/WEC-ES/Dev_passages.json")
    test_passages = io_utils.read_passages_file("resources/WEC-ES/Test_passages.json")
    print("Train Passages=" + str(len(train_passages)))
    print("Dev Passages=" + str(len(dev_passages)))
    print("Test Passages=" + str(len(test_passages)))

    train_clusters = io_utils.read_gold_file("resources/WEC-ES/Train_gold_clusters.json")
    dev_clusters = io_utils.read_gold_file("resources/WEC-ES/Dev_gold_clusters.json")
    test_clusters = io_utils.read_gold_file("resources/WEC-ES/Test_gold_clusters.json")
    print("Train Clusters=" + str(len(train_clusters)))
    print("Dev Clusters=" + str(len(dev_clusters)))
    print("Test Clusters=" + str(len(test_clusters)))

    # query_lengths = [len(exm["context"]) for exm in train_query_examples.values()]
    # passages_lengths = [len(exm["context"]) for exm in train_passages.values()]
    # print("max query size = " + str(max(query_lengths)))
    # print("max passage size = " + str(max(passages_lengths)))
    #
    # print("avg query size = " + str(sum(query_lengths) / len(query_lengths)))
    # print("avg passage size = " + str(sum(passages_lengths) / len(passages_lengths)))
