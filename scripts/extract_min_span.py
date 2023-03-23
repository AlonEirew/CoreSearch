# read json file into a list of dictionaries
import json
from typing import List

import nltk
import stanza
import spacy

# nltk.download('averaged_perceptron_tagger')
# nltk.download('words')
# nltk.download('maxent_ne_chunker')
# stanza_inst = stanza.Pipeline(lang='en', processors='tokenize,pos,ner')


def create_new_ment(tag_type, result_spacy, index):
    last_index_added = -1
    new_ment = list()
    for i, triplet in enumerate(result_spacy):
        if triplet[index] == tag_type and "NUM" not in triplet[1] and "DET" not in triplet[1]:
            if last_index_added == -1 or i == last_index_added + 1:
                new_ment.append(triplet[0])
                last_index_added = i
        elif triplet[1] == 'PART' and new_ment and i == last_index_added + 1:
            new_ment.append(triplet[0])
            last_index_added = i
    return new_ment


def can_automatic_select(new_ment: List[str], head_ment: List[str]):
    repetitive_ment = ["ceremony", "awards", "attack", "earthquake", "massacre", "massacres", "test", "massacre", "earthquakes",
                       "crash", "disaster", "convention", "fire", "flood", "floods", "tournament", "contest", "bushfires", "bushfire",
                       "tour", "festival", "event", "bombing", "bombings", "edition", "summit", "hijacking", "shooting",
                       "conference", "genocide", "pageant"]
    # ceremony, awards, attack, oscars, earthquake, crash, disaster, convention, fire, tournament, tour, festival, event
    if len(new_ment) == 1 and new_ment[0].lower() in repetitive_ment:
        return new_ment
    elif len(head_ment) == 1 and head_ment[0].lower() in repetitive_ment:
        return head_ment
    else:
        return None


def get_user_response(orig_men: List[str], new_ment: List[str], head_ment: List[str]):
    user_ment = None
    new_ment_str = " ".join(new_ment)
    orig_men_str = " ".join(orig_men)
    head_ment_str = " ".join(head_ment)
    print(f"Enter to continue (1=[{new_ment_str}], 2=[{head_ment_str}], 3=[{orig_men_str}], 4=Remove (Not Mention), 5=take a break, write a new mention): ")
    # can_auto = can_automatic_select(new_ment, head_ment)
    can_auto = False
    if can_auto:
        return can_auto
    else:
        while not user_ment:
            user_ment = input("Enter your choice:")
        if user_ment == '1':
            return new_ment
        elif user_ment == '2':
            return head_ment
        elif user_ment == '3':
            return orig_men
        elif user_ment == '4':
            return None
        elif user_ment == '5':
            write_json()
            exit()
        else:
            return user_ment.split()
    return new_ment


def fix_mention_indexes(mentions:List[object], new_ment:List[str]):
    for ment in mentions:
        new_ment_toks = list()
        ment_toks = ment['tokens_number']
        ment['tokens_number_original'] = ment_toks
        ment['tokens_str_original'] = ment['tokens_str']
        ment_str = ment['mention_context'][ment_toks[0]:ment_toks[-1] + 1]
        aligned_ment = list(zip(ment_str, ment_toks))
        for i, tup in enumerate(aligned_ment):
            if tup[0] == new_ment[0]:
                j = i
                k = 0
                while k < len(new_ment) and aligned_ment[j][0] == new_ment[k]:
                    new_ment_toks.append(aligned_ment[j])
                    j += 1
                    k += 1
                break

        new_tok_str, new_tok_num = zip(*new_ment_toks)
        ment['tokens_number'] = list(new_tok_num)
        ment['tokens_str'] = ' '.join(new_tok_str)
        assert ment['tokens_str'] == ' '.join(new_ment), \
            f"new_ment = {' '.join(new_ment)}, ment['tokens_str'] = {ment['tokens_str']}"
        assert ment['mention_context'][ment['tokens_number'][0]:ment['tokens_number'][-1] + 1] == new_ment, \
            f"new_ment = {' '.join(new_ment)}, ment_context = {ment['mention_context'][ment['tokens_number'][0]:ment['tokens_number'][-1] + 1]}"
        ment['span_fix'] = True


def write_json():
    with open(OUT_FILE, 'w') as f:
        json.dump(DATA, f, default=lambda o: o.__dict__, ensure_ascii=False, indent=4)


def load_ments_into_map():
    mentions_map = dict()
    for mention in DATA:
        if 'span_fix' in mention and mention['span_fix'] is True:
            continue

        if 'remove' in mention and mention['remove'] is True:
            continue

        if len(mention['tokens_number']) == 1:
            continue

        if mention['tokens_str'] not in mentions_map:
            mentions_map[mention['tokens_str']] = list()
        mentions_map[mention['tokens_str']].append(mention)

    return mentions_map


def mark_mentions_for_removal(mentions: List[object]):
    for mention in mentions:
        mention['remove'] = True


def main():
    spacy_inst = spacy.load("en_core_web_trf")
    mentions_map = load_ments_into_map()
    # count the number of elements in the map of lists
    # mentions_left = sum(map(len, mentions_map.values()))
    mentions_left = len(mentions_map)
    for index, ment_str in enumerate(mentions_map):
        # if index == 100:
        #     break
        print(f"mentions_left = {mentions_left}")

        mentions = mentions_map[ment_str]
        # extract pos, ner & noun_chunks using spacy
        doc_spacy = spacy_inst(ment_str)
        result_spacy = [(token.text, token.pos_, token.tag_, token.ent_type_) for token in doc_spacy]
        chunks = [chunk for chunk in doc_spacy.noun_chunks]
        pos_tags = set(map(lambda x: x[1], result_spacy))
        ner_tags = set(map(lambda x: x[3], result_spacy))

        head_ment = list()
        if len(chunks) == 1:
            head_ment = [chunks[0].root.head.text]

        new_ment = list()
        if "VERB" in pos_tags:
            new_ment = create_new_ment("VERB", result_spacy, 1)
        elif "EVENT" in ner_tags:
            new_ment = create_new_ment("EVENT", result_spacy, 3)
        elif "NOUN" in pos_tags:
            new_ment = create_new_ment("NOUN", result_spacy, 1)
        elif "PROPN" in pos_tags:
            new_ment = create_new_ment("PROPN", result_spacy, 1)

        if not new_ment:
            new_ment = ment_str.split()

        print_mention_contexts(mentions)
        print("-------------")
        print(f"before spacy tokenized = {result_spacy}")
        print(f"before spacy = {ment_str}")
        # print(f"ment chunks = {chunks}")
        # if len(chunks) == 1:
        #     print(f"head_ment = {chunks[0].root.head.text}")
        # print(f"new_ment = {' '.join(new_ment)}")
        new_ment = get_user_response(ment_str.split(), new_ment, head_ment)
        print("-------------")
        if not new_ment:
            mark_mentions_for_removal(mentions)
            print(f"mention marked for remove")
        else:
            fix_mention_indexes(mentions, new_ment)
            print(f"final_ment = {' '.join(mentions[0]['mention_context'][mentions[0]['tokens_number'][0]:mentions[0]['tokens_number'][-1] + 1])}")
        print("#######")
        mentions_left -= 1

    print()


def print_mention_contexts(mentions):
    for ment in mentions:
        ineer_ment_context = ment['mention_context']
        print(f"{' '.join(ineer_ment_context[0:ment['tokens_number'][0]])} "
              f"<@@{' '.join(ineer_ment_context[ment['tokens_number'][0]:ment['tokens_number'][-1] + 1])}@@> "
              f"{' '.join(ineer_ment_context[ment['tokens_number'][-1] + 1:-1])}")


if __name__ == '__main__':
    IN_FILE = "/Users/aloneirew/workspace/Datasets/WECEng/Test_Event_gold_mentions_validated_fixed_v7.json"
    OUT_FILE = "/Users/aloneirew/workspace/Datasets/WECEng/Test_Event_gold_mentions_validated_fixed_v8.json"
    with open(IN_FILE) as f:
        DATA = json.load(f)
    try:
        main()
    except Exception as e:
        print(f"AssertionError = {e}")
    finally:
        print(f"Closing and saving current state to file-{OUT_FILE}")
        write_json()

# extract part of speech tags using nltk
# result_pos_nltk = nltk.pos_tag(ment.split())
# print(f"NLTK pos = {result_pos_nltk}")
# extract named entity tags using nltk
# result_ner_nltk = nltk.ne_chunk(ment.split())
# print(f"NLTK ne = {result_ner_nltk}")

# extract pos & ner tags using stanza
# doc_stanza = stanza_inst(ment)
# for sent in doc_stanza.sentences:
#     result_pos_staza = [(word.text, word.pos) for word in sent.words]
#     result_ner_stanza = [(ent.text, ent.type) for ent in sent.ents]
#     print(f"stanza_pos = {result_pos_staza}")
#     print(f"stanza_ner = {result_ner_stanza}")
