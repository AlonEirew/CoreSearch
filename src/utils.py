from typing import Dict, List


def read_queries_file(queries_file):
    return read_id_sent_file(queries_file)


def read_passages_file(passages_file):
    return read_id_sent_file(passages_file)


def read_id_sent_file(in_file:str) -> Dict[str, str]:
    queries = dict()
    with open(in_file, "r") as fis:
        readlines = fis.readlines()
        for line in readlines:
            line_splt = line.strip().split("\t")
            queries[line_splt[0]] = line_splt[1]

    return queries


def read_gold_file(gold_file: str) -> Dict[str, List[str]]:
    gold_labels = dict()
    with open(gold_file, "r") as fis:
        readlines = fis.readlines()
        for line in readlines:
            line_splt = line.strip().split("\t")
            gold_labels[line_splt[0]] = line_splt[1:]

    return gold_labels
