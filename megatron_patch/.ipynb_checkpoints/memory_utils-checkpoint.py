from megatron_patch.memory.trainer_v5 import TrainerMemoryManager
from megatron_patch.memory.inference_manager import InferenceMemoryManager
import torch
def megatron_model(model):
    if isinstance(model, list):
        assert len(model) == 1
        model = model[0]
    else:
        model = model
    return model

def create_trainer_memory_manager(
    model,
    optimizer,
    bucket_size_mb=0,
) :
    """
    Create a trainer memory manager based on megatron version.
    """

        

    cls = TrainerMemoryManager
    
    return cls(
        model,
        optimizer,
        bucket_size_mb,
    )

def create_inference_memory_manager(
    model,
    bucket_size_mb=0,
) :
    """
    Create a trainer memory manager based on megatron version.
    """

        

    cls = InferenceMemoryManager
    
    return cls(
        model,
        bucket_size_mb,
    )
