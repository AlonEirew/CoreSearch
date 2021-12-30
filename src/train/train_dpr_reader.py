from haystack.nodes import FARMReader


def main():
    # qa_model = "distilbert-base-uncased-distilled-squad", "facebook/dpr-reader-single-nq-base", "facebook/dpr-reader-multiset-base"
    qa_model = "deepset/roberta-base-squad2"
    reader = FARMReader(model_name_or_path=qa_model, use_gpu=True)
    reader.train(
        data_dir="data/resources/squad",
        train_filename="Train_squad_format.json",
        dev_filename="Dev_squad_format.json",
        evaluate_every=1300,
        use_gpu=True,
        n_epochs=1,
        save_dir="data/checkpoints/squad_roberta_tmp"
    )

    print("Done!")


if __name__ == '__main__':
    main()
