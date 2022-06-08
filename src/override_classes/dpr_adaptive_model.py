from pathlib import Path
from typing import List, Union, Optional, Callable

import torch
from haystack.modeling.data_handler.processor import Processor

from haystack.modeling.model.adaptive_model import AdaptiveModel
from haystack.modeling.model.language_model import LanguageModel
from haystack.modeling.model.prediction_head import PredictionHead


class DPRAdaptiveModel(AdaptiveModel):
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
        super(DPRAdaptiveModel, self).__init__(language_model, prediction_heads, embeds_dropout_prob,
                                               lm_output_types, device, loss_aggregation_fn)

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
        output_tuple = self.language_model.forward(**kwargs, output_hidden_states=output_hidden_states,
                                                   output_attentions=output_attentions)
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

    def logits_to_loss(self, logits: torch.Tensor, global_step: Optional[int] = None, **kwargs):
        """
        Get losses from all prediction heads & reduce to single loss *per sample*.

        :param logits: Logits, can vary in shape and type, depending on task.
        :param global_step: Number of current training step.
        :param kwargs: Placeholder for passing generic parameters.
                       Note: Contains the batch (as dict of tensors), when called from Trainer.train().
        :return: torch.tensor that is the per sample loss (len: batch_size)
        """
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        # This aggregates the loss per sample across multiple prediction heads
        # Default is sum(), but you can configure any fn that takes [Tensor, Tensor ...] and returns [Tensor]
        loss = self.loss_aggregation_fn(all_losses, global_step=global_step, batch=kwargs)
        return loss

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
            load_dir = kwargs['load_dir']
            device = kwargs['device']
            strict = kwargs['strict']
            model = cls.subclasses["AdaptiveModel"].load(load_dir=load_dir, device=device, strict=strict)
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
