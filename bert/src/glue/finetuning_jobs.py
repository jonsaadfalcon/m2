# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# """Contains GLUE job objects for the simple_glue_trainer."""
import atexit
import copy
import gc
import multiprocessing as mp
import os
import sys
from multiprocessing import managers
from typing import Any, Dict, List, Optional, Union, cast

# Add glue folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
print(sys.path)

import torch
from composer import ComposerModel
from composer.core import Callback
from composer.core.evaluator import Evaluator
from composer.core.types import Dataset
from composer.devices import Device, DeviceGPU
from composer.loggers import LoggerDestination
from composer.optim import ComposerScheduler, DecoupledAdamW
from src.optim.create_param_groups import create_param_groups
from composer.trainer.trainer import Trainer
from composer.utils import dist, reproducibility
from data import create_glue_dataset
from torch.utils.data import DataLoader

import torch.nn as nn
import datasets
from datasets import load_dataset

def _build_dataloader(dataset, **kwargs):
    import transformers
    dataset = cast(Dataset, dataset)

    return DataLoader(
        dataset=dataset,
        sampler=dist.get_sampler(dataset, drop_last=False, shuffle=False),
        collate_fn=transformers.default_data_collator,
        **kwargs,
    )


Metrics = Dict[str, Dict[str, Any]]

TASK_NAME_TO_NUM_LABELS = {
    'contract_nli': 3,
    '20news': 20,
    'ecthr': 9,
    'mnli': 3,
    'rte': 2,
    'mrpc': 2,
    'qnli': 2,
    'qqp': 2,
    'sst2': 2,
    'stsb': 1,
    'cola': 2,
}


def reset_trainer(trainer: Trainer, garbage_collect: bool = False):
    """Cleans up memory usage left by trainer."""
    trainer.close()
    # Unregister engine from atexit to remove ref
    atexit.unregister(trainer.engine._close)
    # Close potentially persistent dataloader workers
    loader = trainer.state.train_dataloader
    if loader and loader._iterator is not None:  # type: ignore
        loader._iterator._shutdown_workers()  # type: ignore
    # Explicitly delete attributes of state as otherwise gc.collect() doesn't free memory
    for key in list(trainer.state.__dict__.keys()):
        delattr(trainer.state, key)
    # Delete the rest of trainer attributes
    for key in list(trainer.__dict__.keys()):
        delattr(trainer, key)
    if garbage_collect:
        gc.collect()
        torch.cuda.empty_cache()


class FineTuneJob:
    """Encapsulates a fine-tuning job.

    Tasks should subclass FineTuneJob and implement the
    get_trainer() method.

    Args:
        name (str, optional): job name. Defaults to the class name.
        load_path (str, optional): path to load checkpoints. Default: None
        save_folder (str, optional): path to save checkpoints. Default: None
        kwargs (dict, optional): additional arguments passed available to the Trainer.
    """

    def __init__(
        self,
        job_name: Optional[str] = None,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        seed: int = 42,
        **kwargs,
    ):
        reproducibility.seed_all(seed)
        self._job_name = job_name
        self.seed = seed
        self.load_path = load_path
        self.save_folder = save_folder
        self.kwargs = kwargs

    def get_trainer(self, device: Optional[Union[str, Device]]) -> Trainer:
        """Returns the trainer for the job."""
        raise NotImplementedError

    def print_metrics(self, metrics: Metrics):
        """Prints fine-tuning results."""
        job_name = self.job_name

        print(f'Results for {job_name}:')
        print('-' * (12 + len(job_name)))
        for eval, metric in metrics.items():
            for metric_name, value in metric.items():
                print(f'{eval}: {metric_name}, {value*100:.2f}')
        print('-' * (12 + len(job_name)))

    @property
    def job_name(self) -> str:
        """Job name, defaults to class name."""
        if self._job_name is not None:
            return self._job_name
        return self.__class__.__name__

    def run(self,
            gpu_queue: Optional[mp.Queue] = None,
            process_to_gpu: Optional[managers.DictProxy] = None
           ) -> Dict[str, Any]:
        """Trains the model, optionally pulling a GPU id from the queue.

        Returns:
            A dict with keys:
            * 'checkpoints': list of saved_checkpoints, if any,
            * 'metrics': nested dict of results, accessed by
                        dataset and metric name, e.g.
                        ``metrics['glue_mnli']['MulticlassAccuracy']``.
        """
        if gpu_queue is None:
            if torch.cuda.device_count() > 0:
                gpu_id = 0
                device = DeviceGPU(gpu_id)
            else:
                gpu_id = None
                device = 'cpu'
        else:
            current_pid = os.getpid()
            assert process_to_gpu is not None
            if current_pid in process_to_gpu:
                gpu_id = process_to_gpu[current_pid]
            else:
                gpu_id = gpu_queue.get()
                process_to_gpu[current_pid] = gpu_id
            device = DeviceGPU(gpu_id)

        print(f'Running {self.job_name} on GPU {gpu_id}')

        trainer = self.get_trainer(device=device)

        #print("Found the trainer!")
        #print(trainer.__dict__)

        """ concat_positional_embeddings = True
        if concat_positional_embeddings:

            positional_embeddings = trainer._original_model.model.bert.embeddings.position_embeddings
            concatenated_output = nn.Embedding(num_embeddings=512, embedding_dim=768, device="cuda:0")
            
            # Copy the weights from the original embedding to the expanded one
            concatenated_output.weight.data[:128] = positional_embeddings.weight.data
            concatenated_output.weight.data[128:256] = positional_embeddings.weight.data
            concatenated_output.weight.data[256:384] = positional_embeddings.weight.data
            concatenated_output.weight.data[384:512] = positional_embeddings.weight.data

            #print("concatenated_output shape")
            #print(concatenated_output)
            trainer._original_model.model.bert.embeddings.position_embeddings = concatenated_output

            
            #print("Print modified model")
            #print(trainer._original_model.model)

            #assert False """

        ############################################################

        trainer.fit()

        collected_metrics: Dict[str, Dict[str, Any]] = {}
        for eval_name, metrics in trainer.state.eval_metrics.items():
            collected_metrics[eval_name] = {
                name: metric.compute().cpu().numpy()
                for name, metric in metrics.items()
            }

        saved_checkpoints = copy.copy(trainer.saved_checkpoints)
        reset_trainer(trainer, garbage_collect=True)

        self.print_metrics(collected_metrics)

        output = {
            'checkpoints': saved_checkpoints,
            'metrics': collected_metrics,
            'job_name': self.job_name
        }

        return output


class GlueClassificationJob(FineTuneJob):

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        task_name: Optional[str] = None,
        num_labels: Optional[int] = -1,
        eval_interval: str = '1000ba',
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 512,
        max_duration: Optional[str] = '3ep',
        batch_size: Optional[int] = 32,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        if task_name is None:
            raise ValueError(
                'GlueClassificationJob should not be instantiated directly. Please instantiate a specific glue job type instead (e.g. MNLIJob).'
            )
        super().__init__(job_name, load_path, save_folder, seed, **kwargs)

        self.task_name = task_name

        self.num_labels = num_labels
        self.eval_interval = eval_interval
        self.tokenizer_name = tokenizer_name
        self.model = model

        self.scheduler = scheduler
        print('Max sequence length', max_sequence_length)
        self.max_sequence_length = max_sequence_length
        self.max_duration = max_duration
        self.batch_size = batch_size
        self.loggers = loggers
        self.callbacks = callbacks
        self.precision = precision

        # These will be set by the subclasses for specific GLUE tasks
        self.train_dataloader = None
        self.evaluators = None
        self.optimizer = None

    def get_trainer(self, device: Optional[Union[Device, str]] = None):

        return Trainer(model=self.model,
                       optimizers=self.optimizer,
                       schedulers=self.scheduler,
                       train_dataloader=self.train_dataloader,
                       eval_dataloader=self.evaluators,
                       eval_interval=self.eval_interval,
                       load_path=self.load_path,
                       save_folder=self.save_folder,
                       max_duration=self.max_duration,
                       seed=self.seed,
                       device_train_microbatch_size='auto'
                       if torch.cuda.device_count() > 0 else None,
                       load_weights_only=True,
                       load_strict_model_weights=False,
                       loggers=self.loggers,
                       callbacks=self.callbacks,
                       python_log_level='ERROR',
                       run_name=self.job_name,
                       load_ignore_keys=['state/model/model.classifier*'],
                       precision=self.precision,
                       device=device,
                       progress_bar=True,
                       log_to_console=False,
                       **self.kwargs)


class MNLIJob(GlueClassificationJob):
    """MNLI."""

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '2300ba',
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 512,
        max_duration: Optional[str] = '3ep',
        batch_size: Optional[int] = 48,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        lr: Optional[float] = 5.0e-05,
        wd: Optional[float] = 5.0e-06,
        optim_name: Optional[str] = 'DecoupledAdamW',
        **kwargs,
    ):
        super().__init__(model=model,
                         tokenizer_name=tokenizer_name,
                         job_name=job_name,
                         seed=seed,
                         task_name='mnli',
                         num_labels=3,
                         eval_interval=eval_interval,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        print(f"\nGLUE task {self.task_name} Details:")
        print('-- lr:', lr)
        print('-- wd:', wd)
        print(f"-- seed: {seed}")
        if optim_name == 'DecoupledAdamW':
            print(f"-- using DecoupledAdamW optimizer")
            self.optimizer = DecoupledAdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)
        else:
            from torch.optim import AdamW
            print(f"-- using AdamW optimizer")
            self.optimizer = AdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': self.tokenizer_name,
            'max_seq_length': self.max_sequence_length,
        }
        print('Max sequence length in MNLI', max_sequence_length, self.max_sequence_length)

        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset,
                                                  **dataloader_kwargs)
        mnli_eval_dataset = create_glue_dataset(split='validation_matched',
                                                **dataset_kwargs)
        mnli_eval_mismatched_dataset = create_glue_dataset(
            split='validation_mismatched', **dataset_kwargs)
        mnli_evaluator = Evaluator(label='glue_mnli',
                                   dataloader=_build_dataloader(
                                       mnli_eval_dataset, **dataloader_kwargs),
                                   metric_names=['MulticlassAccuracy'])
        
        #print("MNLI Evaluator Found")
        #print(type(mnli_eval_dataset))
        #print(mnli_eval_dataset)
        #assert False
        
        mnli_evaluator_mismatched = Evaluator(
            label='glue_mnli_mismatched',
            dataloader=_build_dataloader(mnli_eval_mismatched_dataset,
                                         **dataloader_kwargs),
            metric_names=['MulticlassAccuracy'])
        self.evaluators = [mnli_evaluator, mnli_evaluator_mismatched]


class RTEJob(GlueClassificationJob):
    """RTE."""

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '100ba',
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 512,
        max_duration: Optional[str] = '3ep',
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        lr: Optional[float] = 1.0e-5,
        wd: Optional[float] = 1.0e-6,
        optim_name: Optional[str] = 'DecoupledAdamW',
        **kwargs,
    ):
        super().__init__(model=model,
                         tokenizer_name=tokenizer_name,
                         job_name=job_name,
                         seed=seed,
                         task_name='rte',
                         num_labels=2,
                         eval_interval=eval_interval,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)
        
        if optim_name == 'DecoupledAdamW':
            self.optimizer = DecoupledAdamW(create_param_groups(None, self.model),
                                            lr=lr,
                                            betas=(0.9, 0.98),
                                            eps=1.0e-06,
                                            weight_decay=wd)
        else:
            from torch.optim import AdamW
            self.optimizer = AdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': self.tokenizer_name,
            'max_seq_length': self.max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset,
                                                  **dataloader_kwargs)
        rte_eval_dataset = create_glue_dataset(split='validation',
                                               **dataset_kwargs)
        rte_evaluator = Evaluator(label='glue_rte',
                                  dataloader=_build_dataloader(
                                      rte_eval_dataset, **dataloader_kwargs),
                                  metric_names=['MulticlassAccuracy'])
        self.evaluators = [rte_evaluator]


class QQPJob(GlueClassificationJob):
    """QQP."""

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '2000ba',
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 512,
        max_duration: Optional[str] = '5ep',
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        lr: Optional[float] = 3.0e-5,
        wd: Optional[float] = 3.0e-6,
        optim_name: Optional[str] = 'DecoupledAdamW',
        **kwargs,
    ):
        super().__init__(model=model,
                         tokenizer_name=tokenizer_name,
                         job_name=job_name,
                         seed=seed,
                         task_name='qqp',
                         num_labels=2,
                         eval_interval=eval_interval,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        print(f"QNLI:")
        print('-- lr:', lr)
        print('-- wd:', wd)
        print(f"-- optim_name: {optim_name}")
        if optim_name == 'DecoupledAdamW':
            self.optimizer = DecoupledAdamW(create_param_groups(None, self.model),
                                            lr=lr,
                                            betas=(0.9, 0.98),
                                            eps=1.0e-06,
                                            weight_decay=wd)
        else:
            from torch.optim import AdamW
            self.optimizer = AdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': self.tokenizer_name,
            'max_seq_length': self.max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset,
                                                  **dataloader_kwargs)
        qqp_eval_dataset = create_glue_dataset(split='validation',
                                               **dataset_kwargs)
        qqp_evaluator = Evaluator(
            label='glue_qqp',
            dataloader=_build_dataloader(qqp_eval_dataset, **dataloader_kwargs),
            metric_names=['MulticlassAccuracy', 'BinaryF1Score'])
        self.evaluators = [qqp_evaluator]


class COLAJob(GlueClassificationJob):
    """COLA."""

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '250ba',
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 512,
        max_duration: Optional[str] = '10ep',
        batch_size: Optional[int] = 32,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        lr: Optional[float] = 5.0e-5,
        wd: Optional[float] = 5.0e-6,
        optim_name: Optional[str] = 'DecoupledAdamW',
        **kwargs,
    ):
        super().__init__(model=model,
                         tokenizer_name=tokenizer_name,
                         job_name=job_name,
                         seed=seed,
                         task_name='cola',
                         num_labels=2,
                         eval_interval=eval_interval,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        print('COLA LR', lr)
        print('COLA WD', wd)
        if optim_name == 'DecoupledAdamW':
            self.optimizer = DecoupledAdamW(create_param_groups(None, self.model),
                                            lr=lr,
                                            betas=(0.9, 0.98),
                                            eps=1.0e-06,
                                            weight_decay=wd)
        else:
            from torch.optim import AdamW
            self.optimizer = AdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': self.tokenizer_name,
            'max_seq_length': self.max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset,
                                                  **dataloader_kwargs)
        cola_eval_dataset = create_glue_dataset(split='validation',
                                                **dataset_kwargs)
        cola_evaluator = Evaluator(label='glue_cola',
                                   dataloader=_build_dataloader(
                                       cola_eval_dataset, **dataloader_kwargs),
                                #    metric_names=['MatthewsCorrCoef', 'MulticlassAccuracy', 'BinaryF1Score']
                                   metric_names=['MatthewsCorrCoef']
        )
        self.evaluators = [cola_evaluator]


class MRPCJob(GlueClassificationJob):
    """MRPC."""

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '100ba',
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 512,
        max_duration: Optional[str] = '10ep',
        batch_size: Optional[int] = 32,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        lr: Optional[float] = 8.0e-5,
        wd: Optional[float] = 8.0e-6,
        optim_name: Optional[str] = 'DecoupledAdamW',
        **kwargs,
    ):
        super().__init__(model=model,
                         tokenizer_name=tokenizer_name,
                         job_name=job_name,
                         seed=seed,
                         task_name='mrpc',
                         num_labels=2,
                         eval_interval=eval_interval,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        if optim_name == 'DecoupledAdamW':
            self.optimizer = DecoupledAdamW(create_param_groups(None, self.model),
                                            lr=lr,
                                            betas=(0.9, 0.98),
                                            eps=1.0e-06,
                                            weight_decay=wd)
        else:
            from torch.optim import AdamW
            self.optimizer = AdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': self.tokenizer_name,
            'max_seq_length': self.max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset,
                                                  **dataloader_kwargs)
        mrpc_eval_dataset = create_glue_dataset(split='validation',
                                                **dataset_kwargs)
        mrpc_evaluator = Evaluator(
            label='glue_mrpc',
            dataloader=_build_dataloader(mrpc_eval_dataset,
                                         **dataloader_kwargs),
            # metric_names=['MulticlassAccuracy', 'BinaryF1Score']
            metric_names=['BinaryF1Score']
        )
        self.evaluators = [mrpc_evaluator]


class QNLIJob(GlueClassificationJob):
    """QNLI."""

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '1000ba',
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 512,
        max_duration: Optional[str] = '10ep',
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        lr: Optional[float] = 1.0e-5,
        wd: Optional[float] = 1.0e-6,
        optim_name: Optional[str] = 'DecoupledAdamW',
        **kwargs,
    ):
        super().__init__(model=model,
                         tokenizer_name=tokenizer_name,
                         job_name=job_name,
                         seed=seed,
                         task_name='qnli',
                         num_labels=2,
                         eval_interval=eval_interval,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        print(f"\nGLUE task {self.task_name} Details:")
        print('-- lr:', lr)
        print('-- wd:', wd)
        print(f"-- seed: {seed}")
        if optim_name == 'DecoupledAdamW':
            print(f"-- using DecoupledAdamW optimizer")
            self.optimizer = DecoupledAdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)
        else:
            from torch.optim import AdamW
            print(f"-- using AdamW optimizer")
            self.optimizer = AdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': self.tokenizer_name,
            'max_seq_length': self.max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset,
                                                  **dataloader_kwargs)
        qnli_eval_dataset = create_glue_dataset(split='validation',
                                                **dataset_kwargs)
        qnli_evaluator = Evaluator(label='glue_qnli',
                                   dataloader=_build_dataloader(
                                       qnli_eval_dataset, **dataloader_kwargs),
                                   metric_names=['MulticlassAccuracy'])
        self.evaluators = [qnli_evaluator]


class SST2Job(GlueClassificationJob):
    """SST2."""

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '500ba',
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 512,
        max_duration: Optional[str] = '3ep',
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        lr: Optional[float] = 3.0e-5,
        wd: Optional[float] = 3.0e-6,
        optim_name: Optional[str] = 'DecoupledAdamW',
        **kwargs,
    ):
        super().__init__(model=model,
                         tokenizer_name=tokenizer_name,
                         job_name=job_name,
                         seed=seed,
                         task_name='sst2',
                         num_labels=2,
                         eval_interval=eval_interval,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        print('SST LR', lr)
        print('SST WD', wd)
        if optim_name == 'DecoupledAdamW':
            print(f"-- using DecoupledAdamW optimizer")
            self.optimizer = DecoupledAdamW(create_param_groups(None, self.model),
                                            lr=lr,
                                            betas=(0.9, 0.98),
                                            eps=1.0e-06,
                                            weight_decay=wd)
        else:
            from torch.optim import AdamW
            print(f"-- using AdamW optimizer")
            self.optimizer = AdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': self.tokenizer_name,
            'max_seq_length': self.max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset,
                                                  **dataloader_kwargs)
        sst2_eval_dataset = create_glue_dataset(split='validation',
                                                **dataset_kwargs)
        sst2_evaluator = Evaluator(label='glue_sst2',
                                   dataloader=_build_dataloader(
                                       sst2_eval_dataset, **dataloader_kwargs),
                                   metric_names=['MulticlassAccuracy'])
        self.evaluators = [sst2_evaluator]


class STSBJob(GlueClassificationJob):
    """STSB."""

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '200ba',
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 512,
        max_duration: Optional[str] = '10ep',
        batch_size: Optional[int] = 32,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        lr: Optional[float] = 3.0e-5,
        wd: Optional[float] = 3.0e-6,
        optim_name: Optional[str] = 'DecoupledAdamW',
        **kwargs,
    ):
        super().__init__(model=model,
                         tokenizer_name=tokenizer_name,
                         job_name=job_name,
                         seed=seed,
                         task_name='stsb',
                         num_labels=1,
                         eval_interval=eval_interval,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)
        
        if optim_name == 'DecoupledAdamW':
            self.optimizer = DecoupledAdamW(create_param_groups(None, self.model),
                                            lr=lr,
                                            betas=(0.9, 0.98),
                                            eps=1.0e-06,
                                            weight_decay=wd)
        else:
            from torch.optim import AdamW
            self.optimizer = AdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': self.tokenizer_name,
            'max_seq_length': self.max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset,
                                                  **dataloader_kwargs)
        stsb_eval_dataset = create_glue_dataset(split='validation',
                                                **dataset_kwargs)
        stsb_evaluator = Evaluator(label='glue_stsb',
                                   dataloader=_build_dataloader(
                                       stsb_eval_dataset, **dataloader_kwargs),
                                   metric_names=['SpearmanCorrCoef'])
        self.evaluators = [stsb_evaluator]

        # Hardcoded for STSB due to a bug (Can be removed once torchmetrics fixes https://github.com/Lightning-AI/metrics/issues/1294)
        self.precision = 'fp32'





######################################################################

import os
import json
import datasets
from datasets import load_dataset
import transformers

_task_column_names = {
    '20news': ('text', None),
    'contract_nli': ('input', None),
    'mimic': ('text', None),
    'ecthr': ('text', None),
}

def create_news20_dataset(split):

    OUTPUT_DIR = "datasets/20news/splits/"
    
    print(f"Split: {split}")

    if split == 'train':
        path = os.path.join(OUTPUT_DIR, 'train.json')
    elif 'val' in split or 'dev' in split:
        path = os.path.join(OUTPUT_DIR, 'dev.json')
        split = 'dev'
    elif split == 'test':
        path = os.path.join(OUTPUT_DIR, 'test.json')
    
    dataset = load_dataset(
        'json', 
        data_files=[path]
    )
    dataset = dataset['train']

    label_names = set(dataset['label'])
    label_names = sorted(label_names)
    label_map = {label: i for i, label in enumerate(label_names)}

    print("label_map for news20")
    print(label_map)

    def map_labels(example):
        example['label'] = label_map[example['label']]
        return example
    
    dataset = dataset.map(map_labels)

    return dataset


def create_mimic_dataset(split):
    OUTPUT_DIR = "datasets/mimiciii/0/50"
    
    print(f"Split: {split}")

    if split == 'train':
        path = os.path.join(OUTPUT_DIR, 'train.json')
    elif 'val' in split or 'dev' in split:
        path = os.path.join(OUTPUT_DIR, 'dev.json')
        split = 'dev'
    elif split == 'test':
        path = os.path.join(OUTPUT_DIR, 'test.json')
    
    dataset = load_dataset(
        'json', 
        data_files=[path]
    )
    dataset = dataset['train']

    label2idx_path = os.path.join(OUTPUT_DIR, 'label2idx.json')
    with open(label2idx_path, 'r') as f:
        label_map = json.load(f)

    num_labels = len(label_map)

    def map_labels(example):
        labels = [0 for i in range(num_labels)]
        for label in example['labels']:
            labels[label_map[label]] = 1
        example['label_ids'] = torch.tensor(labels, dtype=torch.long)
        del example['labels']
        return example
    
    dataset = dataset.map(map_labels)

    return dataset

def create_ecthr_dataset(split):
    OUTPUT_DIR = "datasets/ecthr/"
    
    print(f"Split: {split}")

    if split == 'train':
        path = os.path.join(OUTPUT_DIR, 'train.json')
    elif 'val' in split or 'dev' in split:
        path = os.path.join(OUTPUT_DIR, 'dev.json')
        split = 'dev'
    elif split == 'test':
        path = os.path.join(OUTPUT_DIR, 'test.json')
    
    dataset = load_dataset(
        'json', 
        data_files=[path]
    )
    dataset = dataset['train']

    dataset = dataset.rename_column('labels', 'label')

    label_names = set()
    for row in range(len(dataset)):
        assert type(dataset[row]['label']) == list
        #assert len(dataset[row]['label']) == 1
        for label in dataset[row]['label']:
            label_names.add(label)

    #label_names = set(dataset['label'])
    label_names = sorted(label_names)
    label_map = {label: i for i, label in enumerate(label_names)}

    print("label_map for ecthr")
    print(label_map)
    num_labels = len(label_map)

    def map_labels(example):
        labels = [0 for i in range(num_labels)]
        for label in example['label']:
            labels[label_map[label]] = 1
        example['label_ids'] = torch.tensor(labels, dtype=torch.long)
        #example['labels'] = torch.tensor(labels, dtype=torch.long)
        del example['label']
        return example
    
    dataset = dataset.map(map_labels)

    return dataset

def create_contract_nli_dataset(split, max_retries=10):
    download_config = datasets.DownloadConfig(max_retries=max_retries)
    dataset = datasets.load_dataset(
        "tau/scrolls", "contract_nli",
        split=split,
        download_config=download_config,
    )

    # remap 'id' to 'idx'
    dataset = dataset.rename_column('id', 'idx')

    # remap the labels 
    mapping = {
        'Not mentioned': 0, 
        'Entailment': 1, 
        'Contradiction': 2
    }
    def map_labels(example):
        example['output'] = mapping[example['output']]
        return example
    dataset = dataset.map(map_labels)
    dataset = dataset.rename_column('output', 'label')
    return dataset


def create_long_context_dataset(task_name, split, tokenizer_name, max_seq_length, num_workers=8):
    
    if task_name == '20news':
        dataset = create_news20_dataset(split)
    elif task_name == 'mimic':
        dataset = create_mimic_dataset(split)
    elif task_name == 'contract_nli':
        dataset = create_contract_nli_dataset(split)
    elif task_name == "ecthr":
        dataset = create_ecthr_dataset(split)

    text_column_names = _task_column_names[task_name]
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name) 

    def tokenize_function(inp):
        # truncates sentences to max_length or pads them to max_length

        first_half = inp[text_column_names[0]]
        #second_half = inp[
        #    text_column_names[1]] if text_column_names[1] in inp else None
        return tokenizer(
            text=first_half,
            #text_pair=second_half,
            padding='max_length',
            max_length=max_seq_length,
            truncation=True,
        )
    
    assert isinstance(dataset, datasets.Dataset)
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=None if num_workers == 0 else num_workers,
        batch_size=1000,
        new_fingerprint=f'{task_name}-tok-2-{split}-{max_seq_length}',
        load_from_cache_file=True,
    )

    for column in dataset.features:
        if column not in ['label', 'input_ids', 'token_type_ids', 'attention_mask']:
            dataset = dataset.remove_columns([column])

    return dataset

class ContractNLIJob(GlueClassificationJob):
    """ContractNLIJob."""

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '1000ba',
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 512,
        max_duration: Optional[str] = '10ep',
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        lr: Optional[float] = 1.0e-5,
        wd: Optional[float] = 1.0e-6,
        optim_name: Optional[str] = 'DecoupledAdamW',
        **kwargs,
    ):
        super().__init__(model=model,
                         tokenizer_name=tokenizer_name,
                         job_name=job_name,
                         seed=seed,
                         task_name='contract_nli',
                         num_labels=3,
                         eval_interval=eval_interval,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        print(f"\nGLUE task {self.task_name} Details:")
        print('-- lr:', lr)
        print('-- wd:', wd)
        print(f"-- seed: {seed}")
        if optim_name == 'DecoupledAdamW':
            print(f"-- using DecoupledAdamW optimizer")
            self.optimizer = DecoupledAdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)
        else:
            from torch.optim import AdamW
            print(f"-- using AdamW optimizer")
            self.optimizer = AdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': self.tokenizer_name,
            'max_seq_length': self.max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_long_context_dataset(self.task_name, "train", self.tokenizer_name, self.max_sequence_length, num_workers=8)
        self.train_dataloader = _build_dataloader(train_dataset,
                                                  **dataloader_kwargs)
        eval_dataset = create_long_context_dataset(self.task_name, "validation", self.tokenizer_name, self.max_sequence_length, num_workers=8)

        print("Eval_Dataset Generated - " + self.task_name)
        print(eval_dataset)
        

        long_context_evaluator = Evaluator(label=self.task_name,
                                   dataloader=_build_dataloader(
                                       eval_dataset, **dataloader_kwargs),
                                   metric_names=['MulticlassAccuracy'])
        self.evaluators = [long_context_evaluator]



class News20Job(GlueClassificationJob):
    """News20Job."""

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '1000ba',
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 512,
        max_duration: Optional[str] = '10ep',
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        lr: Optional[float] = 1.0e-5,
        wd: Optional[float] = 1.0e-6,
        optim_name: Optional[str] = 'DecoupledAdamW',
        **kwargs,
    ):
        super().__init__(model=model,
                         tokenizer_name=tokenizer_name,
                         job_name=job_name,
                         seed=seed,
                         task_name='20news',
                         num_labels=2,
                         eval_interval=eval_interval,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        print(f"\nGLUE task {self.task_name} Details:")
        print('-- lr:', lr)
        print('-- wd:', wd)
        print(f"-- seed: {seed}")
        if optim_name == 'DecoupledAdamW':
            print(f"-- using DecoupledAdamW optimizer")
            self.optimizer = DecoupledAdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)
        else:
            from torch.optim import AdamW
            print(f"-- using AdamW optimizer")
            self.optimizer = AdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': self.tokenizer_name,
            'max_seq_length': self.max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_long_context_dataset(self.task_name, "train", self.tokenizer_name, self.max_sequence_length, num_workers=8)
        self.train_dataloader = _build_dataloader(train_dataset,
                                                  **dataloader_kwargs)
        eval_dataset = create_long_context_dataset(self.task_name, "validation", self.tokenizer_name, self.max_sequence_length, num_workers=8)

        print("Eval_Dataset Generated - " + self.task_name)
        print(eval_dataset)
        

        long_context_evaluator = Evaluator(label=self.task_name,
                                   dataloader=_build_dataloader(
                                       eval_dataset, **dataloader_kwargs),
                                   metric_names=['MulticlassAccuracy'])
        self.evaluators = [long_context_evaluator]



class ECTHRJob(GlueClassificationJob):
    """ECTHRJob."""

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '1000ba',
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 512,
        max_duration: Optional[str] = '10ep',
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        lr: Optional[float] = 1.0e-5,
        wd: Optional[float] = 1.0e-6,
        optim_name: Optional[str] = 'DecoupledAdamW',
        **kwargs,
    ):
        super().__init__(model=model,
                         tokenizer_name=tokenizer_name,
                         job_name=job_name,
                         seed=seed,
                         task_name='ecthr',
                         num_labels=9,
                         eval_interval=eval_interval,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        print(f"\nGLUE task {self.task_name} Details:")
        print('-- lr:', lr)
        print('-- wd:', wd)
        print(f"-- seed: {seed}")
        if optim_name == 'DecoupledAdamW':
            print(f"-- using DecoupledAdamW optimizer")
            self.optimizer = DecoupledAdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)
        else:
            from torch.optim import AdamW
            print(f"-- using AdamW optimizer")
            self.optimizer = AdamW(create_param_groups(None, self.model),
                                        lr=lr,
                                        betas=(0.9, 0.98),
                                        eps=1.0e-06,
                                        weight_decay=wd)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': self.tokenizer_name,
            'max_seq_length': self.max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_long_context_dataset(self.task_name, "train", self.tokenizer_name, self.max_sequence_length, num_workers=8)
        self.train_dataloader = _build_dataloader(train_dataset,
                                                  **dataloader_kwargs)
        eval_dataset = create_long_context_dataset(self.task_name, "validation", self.tokenizer_name, self.max_sequence_length, num_workers=8)

        print("Eval_Dataset Generated - " + self.task_name)
        print(eval_dataset)
        

        long_context_evaluator = Evaluator(label=self.task_name,
                                   dataloader=_build_dataloader(
                                       eval_dataset, **dataloader_kwargs),
                                   metric_names=['MulticlassAccuracy'])
        self.evaluators = [long_context_evaluator]