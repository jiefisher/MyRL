# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""

import abc
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron.training import get_args
from megatron.core import mpu
from typing import Optional

import logging

def build_dataloader(
    dataset,
    consumed_samples,
    mbs,
    gbs,
    drop_last=True,
    pad_samples_to_global_batch_size=False,
    collate_fn=None,
    load_gbs=True,
    use_random_sampler=True,
):
    """Buld dataloader given an input dataset."""
    args = get_args()

    logging.info(f"Building dataloader with consumed samples: {consumed_samples}")
    # Common parameters for batch sampler creation
    common_params = {
        "total_samples": len(dataset),
        "consumed_samples": consumed_samples,
        "micro_batch_size": mbs,
        "data_parallel_rank": mpu.get_data_parallel_rank(),
        "data_parallel_size": mpu.get_data_parallel_world_size(),
        "drop_last": drop_last,
        "global_batch_size": gbs,
        "pad_samples_to_global_batch_size": pad_samples_to_global_batch_size,
    }

    if use_random_sampler:
        cls = MegatronPretrainingRandomBatchSampler if load_gbs else MegatronPretrainingRandomSampler
        common_params["seed"] = args.seed
    else:
        cls = MegatronPretrainingBatchSampler if load_gbs else MegatronPretrainingSampler
    batch_sampler = cls(**common_params)

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 0 else False,
    )

def build_pretraining_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    elif args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       persistent_workers=True if args.num_workers > 0 else False,
                                       )

class BaseMegatronSampler:
    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool = True,
        global_batch_size: Optional[int] = None,
        rampup_batch_size: Optional[list] = None,
        pad_samples_to_global_batch_size: Optional[bool] = False,
    ) -> None:
        # Sanity checks.
        if total_samples <= 0:
            raise RuntimeError("no sample to consume: {}".format(total_samples))
        if micro_batch_size <= 0:
            raise RuntimeError(f"micro_batch_size size must be greater than 0, but {micro_batch_size}")
        if data_parallel_size <= 0:
            raise RuntimeError(f"data parallel size must be greater than 0, but {data_parallel_size}")
        if data_parallel_rank >= data_parallel_size:
            raise RuntimeError(
                "data_parallel_rank should be smaller than data size, but {} >= {}".format(
                    data_parallel_rank, data_parallel_size
                )
            )
        if global_batch_size is not None and rampup_batch_size is None:
            if global_batch_size % (micro_batch_size * data_parallel_size) != 0:
                raise RuntimeError(
                    f"`global_batch_size` ({global_batch_size}) is not divisible by "
                    f"`micro_batch_size ({micro_batch_size}) x data_parallel_size "
                    f"({data_parallel_size})`"
                )
        if pad_samples_to_global_batch_size and global_batch_size is None:
            raise RuntimeError(
                f"`pad_samples_to_global_batch_size` can be `True` only when "
                f"`global_batch_size` is set to an integer value"
            )

        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last
        self.global_batch_size = global_batch_size
        self.pad_samples_to_global_batch_size = pad_samples_to_global_batch_size

        logging.info(
            f'Instantiating MegatronPretrainingSampler with total_samples: {total_samples} and consumed_samples: {consumed_samples}'
        )

    def __len__(self):
        num_available_samples: int = self.total_samples - self.consumed_samples
        if self.global_batch_size is not None:
            if self.drop_last:
                num_global_batches = num_available_samples // self.global_batch_size
            else:
                num_global_batches = (num_available_samples + self.global_batch_size - 1) // self.global_batch_size
            # return len of dataloader in terms of micro batches to avoid discrepancy between len of dataloader and
            # num of batches fetched (as training step fetches in terms of micro batches)
            return num_global_batches * (self.global_batch_size // self.micro_batch_times_data_parallel_size)
        else:
            return (num_available_samples - 1) // self.micro_batch_times_data_parallel_size + 1

    @abc.abstractmethod
    def __iter__(self): ...


class MegatronPretrainingSampler(BaseMegatronSampler):
    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def _get_padding_indices(self, pad_samples_num):
        return range(-1, -pad_samples_num - 1, -1)

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        indices = range(self.consumed_samples, self.total_samples)
        if (not self.drop_last) and self.pad_samples_to_global_batch_size:
            pad_samples_num = -len(indices) % self.global_batch_size
            pad_indices = self._get_padding_indices(pad_samples_num)
            indices = chain(indices, pad_indices)

        for idx in indices:
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            assert (
                not self.pad_samples_to_global_batch_size
            ), 'with pad_samples_to_global_batch_size all batches should be complete'
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class MegatronCorePretrainingSampler(MegatronPretrainingSampler):
    def _get_padding_indices(self, pad_samples_num):
        return [None] * pad_samples_num


class MegatronPretrainingRandomSampler(BaseMegatronSampler):
    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool = True,
        global_batch_size: Optional[int] = None,
        pad_samples_to_global_batch_size: Optional[bool] = False,
        seed: int = 0,
    ) -> None:
        super().__init__(
            total_samples=total_samples,
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
            global_batch_size=global_batch_size,
            pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
        )
        assert (
            not pad_samples_to_global_batch_size
        ), "`MegatronPretrainingRandomSampler` does not support sample padding"
        if (not drop_last) and self.micro_batch_times_data_parallel_size > 1:
            raise RuntimeError(
                "`MegatronPretrainingRandomSampler` does not support drop_last=False when micro_batch_size * data_parallel_size > 1. \
                  please reduce your MBS and data parallelism to 1 if you want to use drop_last=False, or switch to drop_last=True to avoid this error"
            )
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size
        self.seed = seed

    def __len__(self):
        active_total_samples = self.total_samples - (self.last_batch_size if self.drop_last else 0)
        num_available_samples = active_total_samples - self.consumed_samples % active_total_samples
        if self.global_batch_size is not None:
            if self.drop_last:
                num_global_batches = num_available_samples // self.global_batch_size
            else:
                num_global_batches = (num_available_samples + self.global_batch_size - 1) // self.global_batch_size
            # return len of dataloader in terms of micro batches to avoid discrepancy between len of dataloader and
            # num of batches fetched (as training step fetches in terms of micro batches)
            return num_global_batches * (self.global_batch_size // self.micro_batch_times_data_parallel_size)
        else:
            if self.drop_last:
                return num_available_samples // self.micro_batch_times_data_parallel_size
            else:
                return (num_available_samples - 1) // self.micro_batch_times_data_parallel_size

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch