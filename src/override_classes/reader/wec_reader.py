import logging
import multiprocessing
from pathlib import Path
from typing import List, Optional, Union, Callable

import torch
from haystack.modeling.data_handler.data_silo import DataSilo
from haystack.modeling.model.optimization import initialize_optimizer
from haystack.modeling.training import Trainer
from haystack.modeling.utils import initialize_device_settings, set_all_seeds
from haystack.nodes import FARMReader, BaseReader
from haystack.schema import Document, MultiLabel

from src.override_classes.reader.wec_qa_inferencer import WECQAInferencer
from src.override_classes.reader.wec_squad_processor import WECSquadProcessor
from src.override_classes.retriever.wec_processor import QUERY_SPAN_START, QUERY_SPAN_END

logger = logging.getLogger(__name__)


class WECReader(FARMReader):
    def __init__(
            self,
            model_name_or_path: str,
            model_version: Optional[str] = None,
            context_window_size: int = 150,
            batch_size: int = 50,
            use_gpu: bool = True,
            no_ans_boost: float = 0.0,
            return_no_answer: bool = False,
            top_k: int = 10,
            top_k_per_candidate: int = 3,
            top_k_per_sample: int = 1,
            num_processes: Optional[int] = None,
            max_seq_len: int = 256,
            doc_stride: int = 128,
            progress_bar: bool = True,
            duplicate_filtering: int = 0,
            use_confidence_scores: bool = True,
            proxies=None,
            local_files_only=False,
            force_download=False,
            use_auth_token: Optional[Union[str, bool]] = None,
            add_special_tokens: bool = False,
            replace_prediction_heads: bool = False,
            **kwargs
    ):
        self.add_special_tokens = add_special_tokens

        # save init parameters to enable export of component config as YAML
        self.set_config(
            model_name_or_path=model_name_or_path, model_version=model_version,
            context_window_size=context_window_size,
            batch_size=batch_size, use_gpu=use_gpu, no_ans_boost=no_ans_boost, return_no_answer=return_no_answer,
            top_k=top_k, top_k_per_candidate=top_k_per_candidate, top_k_per_sample=top_k_per_sample,
            num_processes=num_processes, max_seq_len=max_seq_len, doc_stride=doc_stride, progress_bar=progress_bar,
            duplicate_filtering=duplicate_filtering, proxies=proxies, local_files_only=local_files_only,
            force_download=force_download, use_confidence_scores=use_confidence_scores, **kwargs
        )
        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)

        self.return_no_answers = return_no_answer
        self.top_k = top_k
        self.top_k_per_candidate = top_k_per_candidate
        self.inferencer = WECQAInferencer.load(model_name_or_path, batch_size=batch_size, gpu=use_gpu,
                                               task_type="question_answering", max_seq_len=max_seq_len,
                                               doc_stride=doc_stride, num_processes=num_processes,
                                               revision=model_version,
                                               disable_tqdm=not progress_bar,
                                               strict=False,
                                               proxies=proxies,
                                               local_files_only=local_files_only,
                                               force_download=force_download,
                                               devices=self.devices,
                                               use_auth_token=use_auth_token,
                                               add_special_tokens=self.add_special_tokens,
                                               replace_prediction_heads=replace_prediction_heads,
                                               **kwargs)
        self.inferencer.model.prediction_heads[0].context_window_size = context_window_size
        self.inferencer.model.prediction_heads[0].no_ans_boost = no_ans_boost
        self.inferencer.model.prediction_heads[0].n_best = top_k_per_candidate + 1  # including possible no_answer
        try:
            self.inferencer.model.prediction_heads[0].n_best_per_sample = top_k_per_sample
        except:
            logger.warning("Could not set `top_k_per_sample` in FARM. Please update FARM version.")
        try:
            self.inferencer.model.prediction_heads[0].duplicate_filtering = duplicate_filtering
        except:
            logger.warning("Could not set `duplicate_filtering` in FARM. Please update FARM version.")
        self.max_seq_len = max_seq_len
        self.use_gpu = use_gpu
        self.progress_bar = progress_bar
        self.use_confidence_scores = use_confidence_scores

        if self.add_special_tokens and QUERY_SPAN_START not in self.inferencer.processor.tokenizer.additional_special_tokens:
            special_tokens_dict = {'additional_special_tokens': [QUERY_SPAN_START, QUERY_SPAN_END]}
            self.inferencer.processor.tokenizer.add_special_tokens(special_tokens_dict)
            self.inferencer.model.language_model.model.resize_token_embeddings(len(self.inferencer.processor.tokenizer))

    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None,
            labels: Optional[MultiLabel] = None, add_isolated_node_eval: bool = False):  # type: ignore
        # if isinstance(query, dict):
        #     query = query["query"]

        self.query_count += 1
        if documents:
            predict = self.timing(self.predict, "query_time")
            results = predict(query=query, documents=documents, top_k=top_k)
        else:
            results = {"answers": []}

        # Add corresponding document_name and more meta data, if an answer contains the document_id
        results["answers"] = [BaseReader.add_doc_meta_data_to_answer(documents=documents, answer=answer) for answer in results["answers"]]

        # run evaluation with labels as node inputs
        if add_isolated_node_eval and labels is not None:
            relevant_documents = [label.document for label in labels.labels]
            results_label_input = predict(query=query, documents=relevant_documents, top_k=top_k)

            # Add corresponding document_name and more meta data, if an answer contains the document_id
            results["answers_isolated"] = [BaseReader.add_doc_meta_data_to_answer(documents=documents, answer=answer) for answer in results_label_input["answers"]]

        return results, "output_1"

    def _training_procedure(
        self,
        data_dir: str,
        train_filename: str,
        dev_filename: Optional[str] = None,
        test_filename: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        batch_size: int = 10,
        n_epochs: int = 2,
        learning_rate: float = 1e-5,
        max_seq_len: Optional[int] = None,
        warmup_proportion: float = 0.2,
        dev_split: float = 0,
        evaluate_every: int = 300,
        save_dir: Optional[str] = None,
        num_processes: Optional[int] = None,
        use_amp: str = None,
        checkpoint_root_dir: Path = Path("model_checkpoints"),
        checkpoint_every: Optional[int] = None,
        checkpoints_to_keep: int = 3,
        teacher_model: Optional["FARMReader"] = None,
        teacher_batch_size: Optional[int] = None,
        caching: bool = False,
        cache_path: Path = Path("cache/data_silo"),
        distillation_loss_weight: float = 0.5,
        distillation_loss: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "kl_div",
        temperature: float = 1.0,
        tinybert: bool = False,
    ):
        if dev_filename:
            dev_split = 0

        if num_processes is None:
            num_processes = multiprocessing.cpu_count() - 1 or 1

        set_all_seeds(seed=42)

        # For these variables, by default, we use the value set when initializing the FARMReader.
        # These can also be manually set when train() is called if you want a different value at train vs inference
        if use_gpu is None:
            use_gpu = self.use_gpu
        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        devices, n_gpu = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)

        if not save_dir:
            save_dir = f"../../saved_models/{self.inferencer.model.language_model.name}"
            if tinybert:
                save_dir += "_tinybert_stage_1"

        # 1. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
        label_list = ["start_token", "end_token"]
        metric = "squad"
        processor = WECSquadProcessor(
            tokenizer=self.inferencer.processor.tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metric=metric,
            train_filename=train_filename,
            dev_filename=dev_filename,
            dev_split=dev_split,
            test_filename=test_filename,
            data_dir=Path(data_dir),
            add_special_tokens=self.add_special_tokens
        )
        data_silo: DataSilo

        data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False, max_processes=num_processes, caching=caching, cache_path=cache_path)

        # 3. Create an optimizer and pass the already initialized model
        model, optimizer, lr_schedule = initialize_optimizer(
            model=self.inferencer.model,
            # model=self.inferencer.model,
            learning_rate=learning_rate,
            schedule_opts={"name": "LinearWarmup", "warmup_proportion": warmup_proportion},
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=n_epochs,
            device=devices[0],
            use_amp=use_amp,
        )
        # 4. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time

        trainer = Trainer.create_or_load_checkpoint(
            model=model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=devices[0],
            use_amp=use_amp,
            disable_tqdm=not self.progress_bar,
            checkpoint_root_dir=Path(checkpoint_root_dir),
            checkpoint_every=checkpoint_every,
            checkpoints_to_keep=checkpoints_to_keep,
        )

        # 5. Let it grow!
        self.inferencer.model = trainer.train()
        self.save(Path(save_dir))