import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Union

import torch
from haystack import Document
from haystack.modeling.model.biadaptive_model import BiAdaptiveModel
from haystack.modeling.model.prediction_head import TextSimilarityHead
from tqdm import tqdm
from transformers import AdamW, BertConfig, BertModel, BertTokenizer

from src.data_obj import Query, Passage, Cluster, TrainExample
from src.override_classes.wec_encoders import WECQuestionEncoder, WECContextEncoder
from src.override_classes.wec_text_processor import WECSimilarityProcessor
from src.utils.tokenization import Tokenization


def load_json_file(json_file: str):
    assert json_file
    with open(json_file, "r") as fis:
        return json.load(fis)


def write_json(out_file, obj_to_write):
    with open(out_file, 'w+') as output:
        json.dump(obj_to_write, output, default=lambda o: o.__dict__, indent=4, sort_keys=True, ensure_ascii=False)


def read_train_example_file(train_exp_file: str) -> List[TrainExample]:
    examples_json = load_json_file(train_exp_file)
    train_exmpl = list()
    for train_exp_obj in tqdm(examples_json, desc="Reading Train Exp"):
        train_exmpl.append(TrainExample(train_exp_obj))
    return train_exmpl


def read_query_file(queries_file: str) -> List[Query]:
    queries_json = load_json_file(queries_file)
    queries = list()
    for query_obj in tqdm(queries_json, desc="Reading queries"):
        queries.append(Query(query_obj))
    return queries


def read_passages_file_filtered(passages_file: str, fileter_list: List[str]) -> List[Passage]:
    passages_json = load_json_file(passages_file)
    passages = list()
    for pass_obj in tqdm(passages_json, desc="Reading passages"):
        if pass_obj["id"] in fileter_list:
            passages.append(Passage(pass_obj))
    return passages


def read_passages_file(passages_file: str) -> List[Passage]:
    passages_json = load_json_file(passages_file)
    passages = list()
    for pass_obj in tqdm(passages_json, desc="Reading passages"):
        passages.append(Passage(pass_obj))
    return passages


def read_gold_file(gold_file: str) -> List[Cluster]:
    gold_json = load_json_file(gold_file)
    golds = list()
    for gold_obj in gold_json:
        golds.append(Cluster(gold_obj))

    return golds


def read_id_sent_file(in_file: str) -> Dict[str, str]:
    queries = dict()
    with open(in_file, "r") as fis:
        readlines = fis.readlines()
        for line in readlines:
            line_splt = line.strip().split("\t")
            queries[line_splt[0]] = line_splt[1]

    return queries


def save_checkpoint(path: str, epoch: int, model, tokenization: Tokenization, optimizer: AdamW):
    model_dir = Path(os.path.join(path, "model-{}".format(epoch)))
    query_tokenizer = Path.joinpath(model_dir, Path("query_tokenizer"))
    passage_tokenizer = Path.joinpath(model_dir, Path("passage_tokenizer"))
    print(f"Saving a checkpoint to {model_dir}...")
    if not os.path.exists(query_tokenizer):
        os.makedirs(query_tokenizer)
    if not os.path.exists(passage_tokenizer):
        os.makedirs(passage_tokenizer)

    torch.save(model, Path(os.path.join(model_dir, "language_model.bin")))
    tokenization.query_tokenizer.save_pretrained(query_tokenizer)
    tokenization.passage_tokenizer.save_pretrained(passage_tokenizer)


def save_checkpoint_dpr(path: str, epoch: int, model, tokenization: Tokenization, optimizer: AdamW):
    model_dir = Path(os.path.join(path, "model-{}".format(epoch)))
    qencoder_dir = Path.joinpath(model_dir, Path("query_encoder"))
    pencoder_dir = Path.joinpath(model_dir, Path("passage_encoder"))
    print(f"Saving a checkpoint to {model_dir}...")
    if not os.path.exists(qencoder_dir):
        os.makedirs(qencoder_dir)
    if not os.path.exists(pencoder_dir):
        os.makedirs(pencoder_dir)

    query_encoder = copy.deepcopy(model.query_encoder)
    passage_encoder = copy.deepcopy(model.passage_encoder)
    if hasattr(query_encoder, 'module'):
        query_encoder = query_encoder.module
    if hasattr(passage_encoder, 'module'):
        passage_encoder = passage_encoder.module

    query_encoder_checkpoint = {'epoch': epoch, 'model_state_dict': query_encoder.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}
    passage_encoder_checkpoint = {'epoch': epoch, 'model_state_dict': passage_encoder.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict()}

    query_conf_filename = Path(qencoder_dir) / "language_model_config.json"
    passage_conf_filename = Path(pencoder_dir) / "language_model_config.json"

    setattr(passage_encoder.config, "name", "DPRContextEncoder")
    setattr(passage_encoder.config, "language", "english")
    setattr(query_encoder.config, "name", "DPRQuestionEncoder")
    setattr(query_encoder.config, "language", "english")
    with open(query_conf_filename, "w") as q_conf_file:
        string = query_encoder.config.to_json_string()
        q_conf_file.write(string)
    with open(passage_conf_filename, "w") as p_conf_file:
        string = passage_encoder.config.to_json_string()
        p_conf_file.write(string)

    torch.save(query_encoder_checkpoint, Path(os.path.join(qencoder_dir, "language_model.bin")))
    tokenization.query_tokenizer.save_pretrained(qencoder_dir)
    torch.save(passage_encoder_checkpoint, Path(os.path.join(pencoder_dir, "language_model.bin")))
    tokenization.query_tokenizer.save_pretrained(pencoder_dir)


def load_checkpoint(path):
    model = torch.load(Path(os.path.join(path, "language_model.bin")))
    query_tokenizer = BertTokenizer.from_pretrained(os.path.join(path, "query_tokenizer"))
    passage_tokenizer = BertTokenizer.from_pretrained(os.path.join(path, "passage_tokenizer"))
    return model, query_tokenizer, passage_tokenizer


def read_wec_to_haystack_doc_list(passages_file: str) -> List[Document]:
    passage_dict = read_passages_file(passages_file)
    documents: List[Document] = []
    for passage in tqdm(passage_dict, desc="Converting passages"):
        meta: Dict[str, Any] = dict()
        meta["mention"] = " ".join(passage.mention)
        meta["startIndex"] = passage.startIndex
        meta["endIndex"] = passage.endIndex
        meta["goldChain"] = passage.goldChain
        documents.append(
            Document(
                content=" ".join(passage.context),
                id=passage.id,
                meta=meta
            )
        )

    return documents


def replace_loaded_retriever_model(retriever, model_path, max_seq_len_query, max_seq_len_passage):
    print("Replacing retriever model..")
    model, query_tokenizer, passage_tokenizer = load_checkpoint(model_path)
    prediction_head = TextSimilarityHead(similarity_function="dot_product",
                                         global_loss_buffer_size=150000)
    retriever.passage_encoder = model.passage_encoder
    retriever.passage_tokenizer = passage_tokenizer
    retriever.query_encoder = model.query_encoder
    retriever.query_tokenizer = query_tokenizer

    retriever.model = BiAdaptiveModel(
        language_model1=model.query_encoder,
        language_model2=model.passage_encoder,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm1_output_types=["per_sequence"],
        lm2_output_types=["per_sequence"],
        device=str(retriever.devices[0]),
    )

    retriever.processor = WECSimilarityProcessor(query_tokenizer=retriever.query_tokenizer,
                                                 passage_tokenizer=retriever.passage_tokenizer,
                                                 max_seq_len_passage=max_seq_len_passage,
                                                 max_seq_len_query=max_seq_len_query,
                                                 label_list=["hard_negative", "positive"],
                                                 metric="text_similarity_metric",
                                                 embed_title=False,
                                                 num_hard_negatives=0,
                                                 num_positives=1,
                                                 tokenization=Tokenization(query_tokenizer=retriever.query_tokenizer,
                                                                           passage_tokenizer=retriever.passage_tokenizer,
                                                                           max_query_size=max_seq_len_query,
                                                                           max_passage_size=max_seq_len_passage))


def replace_retriever_model(retriever, query_encoder, passage_encoder, query_tokenizer, passage_tokenizer,
                            max_seq_len_query, max_seq_len_passage):
    print("Replacing retriever model..")
    wec_query_encoder = WECQuestionEncoder(query_encoder)
    wec_passage_encoder = WECContextEncoder(passage_encoder)
    tokenization = Tokenization(query_tok_file=query_tokenizer,
                                passage_tok_file=passage_tokenizer,
                                max_query_size=max_seq_len_query,
                                max_passage_size=max_seq_len_passage)

    prediction_head = TextSimilarityHead(similarity_function="dot_product",
                                         global_loss_buffer_size=150000)
    retriever.passage_encoder = wec_passage_encoder
    retriever.passage_tokenizer = tokenization.passage_tokenizer
    retriever.query_encoder = wec_query_encoder
    retriever.query_tokenizer = tokenization.query_tokenizer

    retriever.model = BiAdaptiveModel(
        language_model1=retriever.query_encoder,
        language_model2=retriever.passage_encoder,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm1_output_types=["per_sequence"],
        lm2_output_types=["per_sequence"],
        device=str(retriever.devices[0]),
    )

    retriever.processor = WECSimilarityProcessor(query_tokenizer=retriever.query_tokenizer,
                                                 passage_tokenizer=retriever.passage_tokenizer,
                                                 max_seq_len_passage=max_seq_len_passage,
                                                 max_seq_len_query=max_seq_len_query,
                                                 label_list=["hard_negative", "positive"],
                                                 metric="text_similarity_metric",
                                                 embed_title=False,
                                                 num_hard_negatives=0,
                                                 num_positives=1,
                                                 tokenization=tokenization)


def load_model_bkp(pretrained_model_name_or_path: Union[Path, str], **kwargs):
    haystack_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
    if os.path.exists(haystack_lm_config):
        # Haystack style
        bert_config = BertConfig.from_pretrained(haystack_lm_config)
        haystack_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
        model = BertModel.from_pretrained(haystack_lm_model, config=bert_config, **kwargs)
    else:
        # Pytorch-transformer Style
        model = BertModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
    return model
