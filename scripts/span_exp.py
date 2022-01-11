from typing import List

from src.data_obj import Passage, Cluster
from src.utils import io_utils


def main():
    cluster_dict = load_res()
    total_cases = 0
    total_clust = set()
    for cluster in cluster_dict:
        answers = cluster["answers"]
        contexts = cluster["contexts"]
        for ctx in contexts:
            for ans in answers:
                contain = True
                ans = list(ans)
                ctx_cp = ctx["text"].copy()
                ctx_cp[ctx["answer"]["start"]] = "-"
                ctx_cp[ctx["answer"]["end"]] = "-"
                while contain:
                    contain = contains(ans, ctx_cp)
                    if contain:
                        if contain[0] >= ctx["answer"]["start"] and contain[1] - 1 <= ctx["answer"]["end"]:
                            # If bounded within the gold span, skip.
                            ctx_cp[contain[0]] = "-"
                            ctx_cp[contain[1] - 1] = "-"
                        else:
                            total_cases += 1
                            total_clust.add(cluster["id"])
                            new_cp = ctx_cp.copy()
                            new_cp.insert(contain[0], "<")
                            new_cp.insert(contain[1] + 1, ">")
                            ctx_span = "(" + str(contain[0]) + "," + str(contain[1] - 1) + ")"
                            ctx_gold_span = "(" + str(ctx["answer"]["start"]) + "," + str(ctx["answer"]["end"]) + ")"
                            print("Found:")
                            # print("\tAnswer MentionId: " + ans["id"] + "")
                            print("\tAnswer ClusterTitle: " + cluster["title"] + "")
                            print("\t\tAnswer Text: " + " ".join(ans))
                            print("\tIn contextId: " + ctx["id"])
                            print("\t\tContext Text:" + " ".join(new_cp))
                            print("\t\tContext Gold Ment:" + " ".join(ctx["answer"]["mention"]))
                            print("\t\tContext Gold Span:" + ctx_gold_span)
                            print("\t\tIn Position: " + ctx_span)
                            ctx_cp[contain[0]] = "-"
                            ctx_cp[contain[1] - 1] = "-"

    print("Total such cases=" + str(total_cases))
    print("from unique clusters=" + str(len(total_clust)))


def load_res():
    passages_file = "../data/resources/train/Dev_training_passages.json"
    gold_cluster_file = "../data/resources/WEC-ES/Dev_gold_clusters.json"
    dev_golds: List[Cluster] = io_utils.read_gold_file(gold_cluster_file)
    passage_examples: List[Passage] = io_utils.read_passages_file(passages_file)
    passage_dict = {obj.id: obj for obj in passage_examples}
    clusters_dict = list()
    for cluster in dev_golds:
        cluster_dict = dict()
        cluster_dict["id"] = cluster.cluster_id
        cluster_dict["title"] = cluster.cluster_title
        cluster_dict["answers"] = set()
        cluster_dict["contexts"] = list()
        for mention_id in cluster.mention_ids:
            answer = dict()
            contexts = dict()
            answer["id"] = mention_id
            answer["start"] = passage_dict[mention_id].startIndex
            answer["end"] = passage_dict[mention_id].endIndex
            answer["mention"] = passage_dict[mention_id].mention
            cluster_dict["answers"].add(tuple(answer["mention"]))

            contexts["id"] = mention_id
            contexts["text"] = passage_dict[mention_id].context
            contexts["answer"] = answer
            cluster_dict["contexts"].append(contexts)

        clusters_dict.append(cluster_dict)

    print("Done Loading")
    return clusters_dict


def contains(small, big):
    for i in range(len(big) - len(small) + 1):
        for j in range(len(small)):
            if big[i + j] != small[j]:
                break
        else:
            return i, i + len(small)
    return False


if __name__ == '__main__':
    main()
