"""
Reader training script

Usage:
    train_reader.py [--note=<ExperimentSummary>] --predicting_head=<PredictionHeadName> --doc_dir=<SQuADDirectory> --train_filename=<SQuADTrainFile> --dev_filename=<SQuADDevFile> --checkpoint_dir=<CheckpointDir> --output_model=<ModelOutName> --add_special_tokens=<AddBoundToks> --n_epochs=<Epochs> --max_seq_len=<MaxSequence> --max_seq_len_query=<MaxQuery> --batch_size=<BatchSize> --reader_model=<QueryModel> --evaluate_every=<EvaluationEverySteps> --num_processes=<NumProcesses>

Options:
    -h --help                                   Show this screen.
    --note=<ExperimentSummary>                  Brief note about the experiment
    --predicting_head=<PredictionHeadName>      Predicting head to use (kenton / dpr)
    --doc_dir=<SQuADDirectory>                  SQuAD files directory
    --train_filename=<SQuADTrainFile>           SQuAD train file
    --dev_filename=<SQuADDevFile>               SQuAD development file
    --checkpoint_dir=<CheckpointDir>            Checkpoints directory
    --output_model=<ModelOutName>               Model directory
    --add_special_tokens=<AddBoundToks>         (true/false) whether to add span boundary tokens
    --n_epochs=<Epochs>                         Number of epochs to run
    --max_seq_len=<MaxSequence>                 Max sequence length (query+passage)
    --max_seq_len_query=<MaxQuery>              Max query size
    --batch_size=<BatchSize>                    Batch size
    --reader_model=<QueryModel>                 Reader model to fine-tune
    --evaluate_every=<EvaluationEverySteps>     Number of steps to run evaluation
    --num_processes=<NumProcesses>              Number of processes to use for reading files
"""
from docopt import docopt

from src.override_classes.reader.wec_reader import WECReader


def main():
    doc_dir = _arguments.get("--doc_dir")
    train_filename = _arguments.get("--train_filename")
    dev_filename = _arguments.get("--dev_filename")
    reader_model = _arguments.get("--reader_model")
    checkpoint_dir = _arguments.get("--checkpoint_dir")
    output_model = _arguments.get("--output_model")
    add_special_tokens = True if _arguments.get("--add_special_tokens").lower() == 'true' else False
    max_seq_len = int(_arguments.get("--max_seq_len"))
    max_seq_len_query = int(_arguments.get("--max_seq_len_query"))
    predicting_head = _arguments.get("--predicting_head")
    evaluate_every = int(_arguments.get("--evaluate_every"))
    num_processes = int(_arguments.get("--num_processes"))
    n_epochs = int(_arguments.get("--n_epochs"))
    batch_size = int(_arguments.get("--batch_size"))

    if predicting_head not in ["kenton", "dpr"]:
        raise ValueError("Supported predicting head = (kenton / dpr)")

    save_dir = checkpoint_dir + output_model

    reader = WECReader(model_name_or_path=reader_model, use_gpu=True, max_seq_len=max_seq_len,
                       max_query_length=max_seq_len_query, num_processes=num_processes,
                       add_special_tokens=add_special_tokens, replace_prediction_heads=True,
                       prediction_head_str=predicting_head)
    reader.train(
        data_dir=doc_dir,
        train_filename=train_filename,
        dev_filename=dev_filename,
        evaluate_every=evaluate_every,
        use_gpu=True,
        n_epochs=n_epochs,
        batch_size=batch_size,
        num_processes=num_processes,
        save_dir=save_dir
    )

    print("Done!")


if __name__ == '__main__':
    _arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    print(_arguments)
    main()
