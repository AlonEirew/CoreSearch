from src.override_classes.reader.wec_reader import WECReader


def main():
    """
    qa_models deepset/roberta-base-squad2,
                distilbert-base-uncased-distilled-squad,
                facebook/dpr-reader-single-nq-base,
                facebook/dpr-reader-multiset-base,
                roberta-base
    """
    exper_name = "Kenton"
    qa_model = "roberta-base"
    add_special_tokens = True
    # replace_prediction_heads: Can be one of: {corefqa{NA} / kenton / dpr}
    prediction_head_str = exper_name.lower()

    num_processes = 10
    evaluate_every = 750
    n_epochs = 5
    batch_size = 24

    reader = WECReader(model_name_or_path=qa_model, use_gpu=True,
                       num_processes=num_processes, add_special_tokens=add_special_tokens,
                       replace_prediction_heads=True, prediction_head_str=prediction_head_str)
    reader.train(
        data_dir="data/resources/squad/context",
        train_filename="Train_squad_format_1pos_23neg.json",
        dev_filename="Dev_squad_format_1pos_23neg.json",
        evaluate_every=evaluate_every,
        use_gpu=True,
        n_epochs=n_epochs,
        batch_size=batch_size,
        num_processes=num_processes,
        save_dir=f"data/checkpoints/Reader-RoBERTa_base_{exper_name}"
    )

    print("Done!")


if __name__ == '__main__':
    main()
