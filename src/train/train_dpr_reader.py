from haystack.nodes import FARMReader

from src.override_classes.reader.wec_reader import WECReader


def main():
    '''
    qa_models deepset/roberta-base-squad2,
                distilbert-base-uncased-distilled-squad,
                facebook/dpr-reader-single-nq-base,
                facebook/dpr-reader-multiset-base
    '''
    qa_model = "deepset/roberta-base-squad2"

    add_special_tokens = True
    reader = WECReader(model_name_or_path=qa_model, use_gpu=True,
                       num_processes=8, add_special_tokens=add_special_tokens)
    reader.train(
        data_dir="data/resources/squad/context",
        train_filename="Train_squad_format.json",
        dev_filename="Dev_squad_format.json",
        evaluate_every=1300,
        use_gpu=True,
        n_epochs=1,
        num_processes=8,
        save_dir="data/checkpoints/squad_roberta_ctx_special"
    )

    print("Done!")


if __name__ == '__main__':
    main()
