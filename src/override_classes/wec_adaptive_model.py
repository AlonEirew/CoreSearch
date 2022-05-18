from pathlib import Path
from typing import List, Union, Optional, Callable, Iterable, Dict

import torch
from haystack.modeling.data_handler.processor import Processor
from haystack.modeling.model.adaptive_model import AdaptiveModel
from haystack.modeling.model.language_model import LanguageModel
from haystack.modeling.model.prediction_head import PredictionHead


class WECAdaptiveModel(AdaptiveModel):
    """
    PyTorch implementation containing all the modelling needed for your NLP task. Combines a language
    model and a prediction head. Allows for gradient flow back to the language model component.
    """
    def __init__(
        self,
        language_model: LanguageModel,
        prediction_heads: List[PredictionHead],
        embeds_dropout_prob: float,
        lm_output_types: Union[str, List[str]],
        device: str,
        loss_aggregation_fn: Optional[Callable] = None,
    ):
        super(WECAdaptiveModel, self).__init__(language_model, prediction_heads, embeds_dropout_prob,
                                               lm_output_types, device, loss_aggregation_fn)

        self.loss_aggregation_fn = self.loss_per_head_sum

    def forward(self, output_hidden_states: bool = False, output_attentions: bool = False, **kwargs):
        """
        Push data through the whole model and returns logits. The data will
        propagate through the language model and each of the attached prediction heads.

        :param kwargs: Holds all arguments that need to be passed to the language model
                       and prediction head(s).
        :param output_hidden_states: Whether to output hidden states
        :param output_attentions: Whether to output attentions
        :return: All logits as torch.tensor or multiple tensors.
        """
        # Run forward pass of language model
        output_tuple = self.language_model.forward(**kwargs, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        if output_hidden_states:
            if output_attentions:
                sequence_output, pooled_output, hidden_states, attentions = output_tuple
            else:
                sequence_output, pooled_output, hidden_states = output_tuple
        else:
            if output_attentions:
                sequence_output, pooled_output, attentions = output_tuple
            else:
                sequence_output, pooled_output = output_tuple
        # Run forward pass of (multiple) prediction heads using the output from above
        all_logits = []
        if len(self.prediction_heads) > 0:
            for head, lm_out in zip(self.prediction_heads, self.lm_output_types):
                # Choose relevant vectors from LM as output and perform dropout
                if lm_out == "per_token":
                    output = self.dropout(sequence_output)
                elif lm_out == "per_sequence" or lm_out == "per_sequence_continuous":
                    output = self.dropout(pooled_output)
                elif (
                    lm_out == "per_token_squad"
                ):  # we need a per_token_squad because of variable metric computation later on...
                    output = self.dropout(sequence_output)
                else:
                    raise ValueError(
                        "Unknown extraction strategy from language model: {}".format(lm_out)
                    )

                # Do the actual forward pass of a single head
                all_logits.append(head(output, **kwargs))
        else:
            # just return LM output (e.g. useful for extracting embeddings at inference time)
            all_logits.append((sequence_output, pooled_output))

        if output_hidden_states:
            if output_attentions:
                return all_logits, hidden_states, attentions
            else:
                return all_logits, hidden_states
        elif output_attentions:
            return all_logits, attentions
        return all_logits

    def logits_to_loss_per_head(self, logits: torch.Tensor, **kwargs):
        """
        Collect losses from each prediction head.

        :param logits: Logits, can vary in shape and type, depending on task.
        :return: The per sample per prediciton head loss whose first two dimensions
                 have length n_pred_heads, batch_size.
        """
        all_losses = []
        for head, logits_for_one_head in zip(self.prediction_heads, logits):
            # check if PredictionHead connected to Processor
            assert hasattr(head, "label_tensor_name"), \
                (f"Label_tensor_names are missing inside the {head.task_name} Prediction Head. Did you connect the model"
                " with the processor through either 'model.connect_heads_with_processor(processor.tasks)'"
                " or by passing the processor to the Adaptive Model?")
            all_losses.append(head.logits_to_loss(logits=logits_for_one_head[0], **kwargs))
            all_losses.append(head.pairs_to_loss(pairs_scores=logits_for_one_head[1], **kwargs))
        return all_losses

    @staticmethod
    def loss_per_head_sum(loss_per_head: Iterable, global_step: Optional[int] = None, batch: Optional[Dict] = None):
        """
        Sums up the loss of each prediction head.

        :param loss_per_head: List of losses.
        """
        # [L(Start) + L(End) + L(Pair)] / 3
        return sum(loss_per_head) / 3

    def logits_to_preds(self, logits: torch.Tensor, **kwargs):
        """
        Get predictions from all prediction heads.

        :param logits: Logits that can vary in shape and type, depending on task.
        :return: A list of all predictions from all prediction heads.
        """
        all_preds = []
        # collect preds from all heads
        for head, logits_for_head in zip(self.prediction_heads, logits):
            preds = head.logits_to_preds(logits=logits_for_head, **kwargs)
            all_preds.append(preds)
        return all_preds

    @classmethod
    def load(cls, **kwargs):
        """
        Load corresponding AdaptiveModel Class(AdaptiveModel/ONNXAdaptiveModel) based on the
        files in the load_dir.

        :param kwargs: Arguments to pass for loading the model.
        :return: Instance of a model.
        """
        replace_prediction_heads = False
        if "replace_prediction_heads" in kwargs:
            replace_prediction_heads = kwargs["replace_prediction_heads"]

        if (Path(kwargs["load_dir"]) / "model.onnx").is_file():
            model = cls.subclasses["ONNXAdaptiveModel"].load(**kwargs)
        elif replace_prediction_heads:
            model = cls.load_from_file(**kwargs)
        else:
            model = cls.subclasses["AdaptiveModel"].load(**kwargs)
        return model

    @classmethod
    def load_from_file(cls, load_dir: Union[str, Path], device: str, strict: bool = True, lm_name: Optional[str] = None,
                       replace_prediction_heads: bool = True, processor: Optional[Processor] = None):

        # Language Model
        if lm_name:
            language_model = LanguageModel.load(load_dir, haystack_lm_name=lm_name)
        else:
            language_model = LanguageModel.load(load_dir)

        # Prediction heads
        _, ph_config_files = cls._get_prediction_head_files(load_dir)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            head = PredictionHead.load(config_file, strict=strict)
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)

        model = cls(language_model, prediction_heads, 0.1, ph_output_type, device)
        if processor:
            model.connect_heads_with_processor(processor.tasks)

        return model
