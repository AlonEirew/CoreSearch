import logging
from typing import Union

from haystack.modeling.model import adaptive_model as am
from haystack.modeling.model.language_model import LanguageModel
from haystack.modeling.model.prediction_head import QuestionAnsweringHead
from src.override_classes.dpr_adaptive_model import DPRAdaptiveModel
from src.override_classes.reader.corefqa_head import CorefQuestionAnsweringHead
from src.override_classes.reader.kenton_head import KentonQuestionAnsweringHead
from src.override_classes.wec_adaptive_model import CorefAdaptiveModel

from src.override_classes.reader.dpr_head import DPRQuestionAnsweringHead

logger = logging.getLogger(__name__)


class WECConverter:
    @staticmethod
    def convert_from_transformers(model_name_or_path, device, revision=None, task_type=None, processor=None,
                                  use_auth_token: Union[bool, str] = None, prediction_head_str: str = "corefqa", **kwargs):
        """
        Load a (downstream) model from huggingface's transformers format. Use cases:
         - continue training in Haystack (e.g. take a squad QA model and fine-tune on your own data)
         - compare models without switching frameworks
         - use model directly for inference

        :param model_name_or_path: local path of a saved model or name of a public one.
                                              Exemplary public names:
                                              - distilbert-base-uncased-distilled-squad
                                              - deepset/bert-large-uncased-whole-word-masking-squad2

                                              See https://huggingface.co/models for full list
        :param device: "cpu" or "cuda"
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str
        :param task_type: One of :
                          - 'question_answering'
                          More tasks coming soon ...
        :param processor: populates prediction head with information coming from tasks
        :type processor: Processor
        :return: AdaptiveModel
        """
        lm = LanguageModel.load(model_name_or_path, revision=revision, use_auth_token=use_auth_token, **kwargs)
        if task_type is None:
            # Infer task type from config
            architecture = lm.model.config.architectures[0]
            if "QuestionAnswering" in architecture:
                task_type = "question_answering"
            else:
                logger.error("Could not infer task type from model config. Please provide task type manually. "
                             "('question_answering' or 'embeddings')")

        if task_type == "question_answering":
            if prediction_head_str == "corefqa":
                ph = CorefQuestionAnsweringHead.load(model_name_or_path, revision=revision, **kwargs)
                adaptive_model = CorefAdaptiveModel(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1,
                                                    lm_output_types="per_token", device=device)
            elif prediction_head_str == "kenton":
                ph = KentonQuestionAnsweringHead.load(model_name_or_path, revision=revision, **kwargs)
                adaptive_model = CorefAdaptiveModel(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1,
                                                    lm_output_types="per_token", device=device)
            elif prediction_head_str == "dpr":
                ph = DPRQuestionAnsweringHead.load(model_name_or_path, revision=revision, **kwargs)
                adaptive_model = DPRAdaptiveModel(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1,
                                                  lm_output_types="per_token", device=device)
            else:
                raise TypeError(f"Not a supported head-{prediction_head_str}")
        elif task_type == "embeddings":
            adaptive_model = CorefAdaptiveModel(language_model=lm, prediction_heads=[], embeds_dropout_prob=0.1,
                                                lm_output_types=["per_token", "per_sequence"], device=device)
        else:
            raise ValueError(f"task_type={task_type}, not supported")

        if processor:
            adaptive_model.connect_heads_with_processor(processor.tasks)

        return adaptive_model