from src import io_utils

if __name__ == '__main__':
    train_query_examples = io_utils.read_query_examples_file("resources/train/wec_es_train_qsent_examples.json")
    print("Done loading examples file")
    train_passages = io_utils.read_passages_file("resources/train/wec_es_train_passages.json")
    print("Done loading passages file")

    query_lengths = [len(exm["context"]) for exm in train_query_examples.values()]
    passages_lengths = [len(exm["context"]) for exm in train_passages.values()]
    print("max query size = " + str(max(query_lengths)))
    print("max passage size = " + str(max(passages_lengths)))

    print("avg query size = " + str(sum(query_lengths) / len(query_lengths)))
    print("avg passage size = " + str(sum(passages_lengths) / len(passages_lengths)))
