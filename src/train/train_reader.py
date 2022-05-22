import os

os.environ["MILVUS2_ENABLED"] = "false"
from src.override_classes.reader.wec_reader import WECReader


def main():
    """
    qa_models deepset/roberta-base-squad2,
                distilbert-base-uncased-distilled-squad,
                facebook/dpr-reader-single-nq-base,
                facebook/dpr-reader-multiset-base
    """
    qa_model = "deepset/roberta-base-squad2"
    add_special_tokens = True
    replace_prediction_heads = True
    num_processes = 8
    evaluate_every = 2 #2200
    n_epochs = 1
    batch_size = 10

    reader = WECReader(model_name_or_path=qa_model, use_gpu=True,
                       num_processes=num_processes, add_special_tokens=add_special_tokens,
                       replace_prediction_heads=replace_prediction_heads)
    reader.train(
        data_dir="data/resources/squad/context",
        train_filename="Train_squad_format_1pos_24neg.json",
        dev_filename="Dev_squad_format_1pos_24neg.json",
        evaluate_every=evaluate_every,
        use_gpu=True,
        n_epochs=n_epochs,
        batch_size=batch_size,
        num_processes=num_processes,
        save_dir="data/checkpoints/deepset_roberta_base_squad2_pairwise"
    )

    print("Done!")


if __name__ == '__main__':
    main()
