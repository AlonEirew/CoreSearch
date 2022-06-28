"""
Retriever training script

Usage:
    train_retriever.py [--note=<ExperimentSummary>] --doc_dir=<DPRDirectory> --train_filename=<DPRTrainFile> --dev_filename=<DPRDevFile> --checkpoint_dir=<CheckpointDir> --output_model=<ModelOutName> --add_special_tokens=<AddBoundToks> --n_epochs=<Epochs> --max_seq_len_query=<MaxQuery> --max_seq_len_passage=<MaxPassage> --batch_size=<BatchSize> --query_model=<QueryModel> --passage_model=<PassageModel> --evaluate_every=<EvaluationEverySteps>

Options:
    -h --help                                   Show this screen.
    --note=<ExperimentSummary>                  Brief note about the experiment
    --doc_dir=<DPRDirectory>                    DPR files directory
    --train_filename=<DPRTrainFile>             DPR train file
    --dev_filename=<DPRDevFile>                 DPR development file
    --checkpoint_dir=<CheckpointDir>            Checkpoints directory
    --output_model=<ModelOutName>               Model directory
    --add_special_tokens=<AddBoundToks>         (true/false) whether to add span boundary tokens
    --n_epochs=<Epochs>                         Number of epochs to run
    --max_seq_len_query=<MaxQuery>              Max query size
    --max_seq_len_passage=<MaxPassage>          Max passage size
    --batch_size=<BatchSize>                    Batch size
    --query_model=<QueryModel>                  Query model to fine-tune
    --passage_model=<PassageModel>              Passage model to fine-tune
    --evaluate_every=<EvaluationEverySteps>     Number of steps to run evaluation
"""

from os import path

from docopt import docopt

from src.override_classes.retriever.wec_dense import WECDensePassageRetriever
from src.override_classes.retriever.wec_context_processor import WECContextProcessor
from src.utils.io_utils import write_json


def train():
    doc_dir = _arguments.get("--doc_dir")
    train_filename = _arguments.get("--train_filename")
    dev_filename = _arguments.get("--dev_filename")
    checkpoint_dir = _arguments.get("--checkpoint_dir")
    output_model = checkpoint_dir + _arguments.get("--output_model")
    add_special_tokens = True if _arguments.get("--add_special_tokens").lower() == 'true' else False
    n_epochs = int(_arguments.get("--n_epochs"))
    max_seq_len_query = int(_arguments.get("--max_seq_len_query"))
    max_seq_len_passage = int(_arguments.get("--max_seq_len_passage"))
    batch_size = int(_arguments.get("--batch_size"))
    query_model = _arguments.get("--query_model")
    passage_model = _arguments.get("--passage_model")
    evaluate_every = int(_arguments.get("--evaluate_every"))

    retriever = WECDensePassageRetriever(document_store=None,
                                         query_embedding_model=query_model,
                                         passage_embedding_model=passage_model,
                                         infer_tokenizer_classes=True,
                                         max_seq_len_query=max_seq_len_query,
                                         max_seq_len_passage=max_seq_len_passage,
                                         batch_size=batch_size, use_gpu=True, embed_title=False,
                                         use_fast_tokenizers=False, processor_type=WECContextProcessor,
                                         add_special_tokens=add_special_tokens)

    retriever.train(data_dir=doc_dir,
                    train_filename=train_filename,
                    dev_filename=dev_filename,
                    test_filename=None,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    save_dir=output_model,
                    evaluate_every=evaluate_every,
                    embed_title=False,
                    num_positives=1,
                    num_hard_negatives=1,
                    max_processes=5)

    write_json(path.join(output_model, "params.json"), _arguments)


if __name__ == "__main__":
    _arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    print(_arguments)
    train()
