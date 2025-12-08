# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference Memery manager for Megatron."""
from typing import Optional, List
import gc
import torch
from torch import nn
from megatron.core.distributed import DistributedDataParallel 
from .flat_tensors import BucketizedFlatTensors, FlatTensors
from megatron.training import print_rank_0


class InferenceMemoryManager:
    """
    Memory manager for Megatron inference modules which provides utilities to free memory when unused.
    """

    def __init__(self, model: List[nn.Module], bucket_size_mb: int=0):
        """Manage memory of inference models which only have model weights.

        Args:
            model (List[nn.Module]): The Megatron model chunks to be managed. 
            Should all be unwrapped.
            bucket_size_mb (int): The bucket size when offloading weights.
        """
        self._model = model

        if any(isinstance(model_chunk, (DistributedDataParallel,)) for model_chunk in model):
            all_types = ','.join([str(type(model_chunk)) for model_chunk in model] )
            raise NotImplementedError(f'Only support model type non-DistributedDataParallel, current type is {all_types}.')

        self._weights_offloaded = False
        self._group_flat_weights: Optional[List[BucketizedFlatTensors]] = None
        self._bucket_size_mb = bucket_size_mb

    def offload_weights(self):
        """
        offload weights
        """
        if self._weights_offloaded:
            log_rank_0('Call offload_weights when already offloaded. Ignore it.')
            return

        if self._group_flat_weights is None:
            dtype_to_params = {}
            for model_chunk in self._model:
                for p in model_chunk.parameters():
                    dtype = p.dtype
                    if dtype not in dtype_to_params:
                        dtype_to_params[dtype] = []
                    dtype_to_params[dtype].append(p)

            self._group_flat_weights = []
            for params in dtype_to_params.values():
                self._group_flat_weights.append(
                    BucketizedFlatTensors(params, primary_store_device='cpu', bucket_size_mb=self._bucket_size_mb)
                )

        for flat_weights in self._group_flat_weights:
            flat_weights.copy_to_primary_store()

        self._weights_offloaded = True

    def onload_weights(self):
        """
        onload weights
        """
        if not self._weights_offloaded:
            log_rank_0('Call onload_weights when already onloaded. Ignore it.')
            return

        for flat_weights in self._group_flat_weights:
            flat_weights.copy_to_gpu_buffer()

        self._weights_offloaded = False

    def offloads(self):
        torch.cuda.synchronize()
        torch.distributed.barrier()

        self.offload_weights()
        
        torch.distributed.barrier()
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()

    def onloads(self):
        torch.cuda.synchronize()
        torch.distributed.barrier()
    
        self.onload_weights()
        
        torch.distributed.barrier()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()