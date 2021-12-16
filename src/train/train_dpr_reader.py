from haystack.reader import FARMReader


def main():
    qa_model = "deepset/roberta-base-squad2"
    # qa_model = "distilbert-base-uncased-distilled-squad"
    reader = FARMReader(model_name_or_path=qa_model, use_gpu=True)
    reader.train(
        data_dir="resources/squad",
        train_filename="Train_squad_format.json",
        evaluate_every=20,
        use_gpu=True,
        n_epochs=2,
        save_dir="checkpoints/squad_roberta_2it")


if __name__ == '__main__':
    main()
