import json
import logging
import os
from abc import ABC
from pathlib import Path
from typing import Union

from haystack.modeling.model.language_model import LanguageModel
from src.override_classes.retriever.search_encoders import CoreSearchContextEncoder, CoreSearchQuestionEncoder

logger = logging.getLogger(__name__)


class OverrideLanguageModel(LanguageModel, ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def load(cls, pretrained_model_name_or_path: Union[Path, str], language: str = None,
             use_auth_token: Union[bool, str] = None, **kwargs):

        n_added_tokens = kwargs.pop("n_added_tokens", 0)
        language_model_class = kwargs.pop("language_model_class", None)
        kwargs["revision"] = kwargs.get("revision", None)
        logger.info("LOADING MODEL")
        logger.info("=============")
        config_file = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(config_file):
            logger.info(f"Model found locally at {pretrained_model_name_or_path}")
            # it's a local directory in Haystack format
            config = json.load(open(config_file))
            language_model = cls.subclasses[config["name"]].load(pretrained_model_name_or_path)
        else:
            logger.info(f"Could not find {pretrained_model_name_or_path} locally.")
            logger.info(f"Looking on Transformers Model Hub (in local cache and online)...")

            if language_model_class == "CoreSearchContextEncoder":
                language_model = CoreSearchContextEncoder.load(pretrained_model_name_or_path,
                                                        use_auth_token=use_auth_token, **kwargs)
            elif language_model_class == "CoreSearchQuestionEncoder":
                language_model = CoreSearchQuestionEncoder.load(pretrained_model_name_or_path,
                                                         use_auth_token=use_auth_token, **kwargs)
            else:
                language_model = None

        if not language_model:
            raise Exception(
                f"Model not found for {pretrained_model_name_or_path}. Either supply the local path for a saved "
                f"model or one of bert/roberta/xlnet/albert/distilbert models that can be downloaded from remote. "
                f"Ensure that the model class name can be inferred from the directory name when loading a "
                f"Transformers' model."
            )
        else:
            logger.info(f"Loaded {pretrained_model_name_or_path}")

        # resize embeddings in case of custom vocab
        if n_added_tokens != 0:
            # TODO verify for other models than BERT
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            vocab_size = model_emb_size + n_added_tokens
            logger.info(
                f"Resizing embedding layer of LM from {model_emb_size} to {vocab_size} to cope with custom vocab.")
            language_model.model.resize_token_embeddings(vocab_size)
            # verify
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            assert vocab_size == model_emb_size

        return language_model
