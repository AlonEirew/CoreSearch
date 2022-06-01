import os
from typing import Optional, Dict, List, Union

import torch

from haystack.modeling.data_handler.processor import InferenceProcessor, Processor
from haystack.modeling.infer import QAInferencer
from haystack.modeling.model.adaptive_model import AdaptiveModel
from haystack.modeling.utils import initialize_device_settings
from src.override_classes.reader.processors.wec_squad_processor import WECSquadProcessor
from src.override_classes.wec_adaptive_model import WECAdaptiveModel
from src.override_classes.wec_converter import WECConverter


class WECQAInferencer(QAInferencer):
    def __init__(
            self,
            model: AdaptiveModel,
            processor: Processor,
            task_type: Optional[str],
            batch_size: int = 4,
            gpu: bool = False,
            name: Optional[str] = None,
            return_class_probs: bool = False,
            extraction_strategy: Optional[str] = None,
            extraction_layer: Optional[int] = None,
            num_processes: Optional[int] = None,
            disable_tqdm: bool = False
    ):
        super(WECQAInferencer, self).__init__(
            model,
            processor,
            task_type,
            batch_size,
            gpu,
            name,
            return_class_probs,
            extraction_strategy,
            extraction_layer,
            num_processes,
            disable_tqdm
        )

    @classmethod
    def load(
            cls,
            model_name_or_path: str,
            revision: Optional[str] = None,
            batch_size: int = 4,
            gpu: bool = False,
            task_type: Optional[str] = None,
            return_class_probs: bool = False,
            strict: bool = True,
            max_seq_len: int = 256,
            doc_stride: int = 128,
            extraction_strategy: Optional[str] = None,
            extraction_layer: Optional[int] = None,
            num_processes: Optional[int] = None,
            disable_tqdm: bool = False,
            tokenizer_class: Optional[str] = None,
            use_fast: bool = True,
            tokenizer_args: Dict = None,
            multithreading_rust: bool = True,
            devices: Optional[List[Union[int, str, torch.device]]] = None,
            use_auth_token: Union[bool, str] = None,
            add_special_tokens: bool = False,
            replace_prediction_heads: bool = False,
            **kwargs
    ):
        if tokenizer_args is None:
            tokenizer_args = {}

        if devices is None:
            devices, n_gpu = initialize_device_settings(use_cuda=gpu, multi_gpu=False)

        name = os.path.basename(model_name_or_path)

        # a) either from local dir
        if os.path.exists(model_name_or_path):
            model = WECAdaptiveModel.load(load_dir=model_name_or_path, device=devices[0], strict=strict,
                                          replace_prediction_heads=replace_prediction_heads)
            if task_type == "embeddings":
                processor = InferenceProcessor.load_from_dir(model_name_or_path)
            else:
                processor = Processor.load_from_dir(model_name_or_path)

            if isinstance(processor, WECSquadProcessor):
                processor.add_special_tokens = add_special_tokens

        # b) or from remote transformers model hub
        else:
            if not task_type:
                raise ValueError("Please specify the 'task_type' of the model you want to load from transformers. "
                                 "Valid options for arg `task_type`:"
                                 "'question_answering'")

            # override model predicting_head with pairwise head
            if replace_prediction_heads:
                model = WECConverter.convert_from_transformers(model_name_or_path,
                                                               revision=revision,
                                                               device=devices[0],
                                                               task_type=task_type,
                                                               processor=None,
                                                               use_auth_token=use_auth_token,
                                                               **kwargs)
            else:
                model = AdaptiveModel.convert_from_transformers(model_name_or_path,
                                                                revision=revision,
                                                                device=devices[0],  # type: ignore
                                                                task_type=task_type,
                                                                use_auth_token=use_auth_token,
                                                                **kwargs)

            processor = WECSquadProcessor.convert_from_transformers(model_name_or_path,
                                                                    revision=revision,
                                                                    task_type=task_type,
                                                                    max_seq_len=max_seq_len,
                                                                    doc_stride=doc_stride,
                                                                    tokenizer_class=tokenizer_class,
                                                                    tokenizer_args=tokenizer_args,
                                                                    use_fast=use_fast,
                                                                    add_special_tokens=add_special_tokens,
                                                                    **kwargs)

        # override processor attributes loaded from config or HF with inferencer params
        processor.max_seq_len = max_seq_len
        processor.multithreading_rust = multithreading_rust
        if hasattr(processor, "doc_stride"):
            assert doc_stride < max_seq_len, "doc_stride is longer than max_seq_len. This means that there will be gaps " \
                                             "as the passage windows slide, causing the model to skip over parts of the document. " \
                                             "Please set a lower value for doc_stride (Suggestions: doc_stride=128, max_seq_len=384) "
            processor.doc_stride = doc_stride

        return cls(
            model,
            processor,
            task_type=task_type,
            batch_size=batch_size,
            gpu=gpu,
            name=name,
            return_class_probs=return_class_probs,
            extraction_strategy=extraction_strategy,
            extraction_layer=extraction_layer,
            num_processes=num_processes,
            disable_tqdm=disable_tqdm
        )
