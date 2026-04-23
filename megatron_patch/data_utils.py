import torch
from collections import UserDict
from megatron.training import get_args, print_rank_0
from megatron.legacy.data.data_samplers import build_pretraining_data_loader,build_dataloader
from megatron.core.utils import divide
from megatron.core import mpu
from functools import partial
from dataclasses import dataclass
from collections.abc import Iterator, Mapping
from typing import Dict, List, Optional, Union, Callable
from typing_extensions import Self
from megatron_patch.distributed import broadcast_2d_tensor_within_mp, broadcast_tensor_within_pp, rebalance_nd_tensor, from_parallel_logits_to_logprobs, broadcast_2d_tensor_within_pp

def get_train_valid_test_num_samples(args):
    """Train/valid/test num samples."""

    
    # Number of train/valid/test samples.
    
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size

    
    # eval_iters = (args.train_iters // args.eval_interval + 1) * \
    #              args.eval_iters
    eval_iters = args.eval_iters
    test_iters = args.eval_iters

    return (
        train_samples,
        eval_iters * args.global_batch_size,
        test_iters * args.global_batch_size,
    )
    
def build_train_valid_test_datasets(args, build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""
    train_valid_test_num_samples = get_train_valid_test_num_samples(args)
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_valid_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_valid_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_valid_test_num_samples[2]))
    return build_train_valid_test_datasets_provider(train_valid_test_num_samples)

def build_train_valid_test_data_loaders(
        args, build_train_valid_test_datasets_provider,collate_fn):
    """Build pretraining data loaders."""

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'Only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)

    # Construct the data pipeline
    if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:
        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            args, build_train_valid_test_datasets_provider)

        # Build dataloders.
        train_dataloader = build_dataloader(
            dataset=train_ds,
            consumed_samples=args.consumed_train_samples,
            mbs=1, #cfg.model.reinforce.rollout_micro_batch_size
            gbs=args.global_batch_size, #cfg.model.reinforce.num_rollout_samples
            collate_fn=collate_fn,
            load_gbs=False,
            )
        if args.skip_train:
            valid_dataloader = build_dataloader(
                dataset=valid_ds,
                consumed_samples=0,
                mbs=1, #cfg.model.reinforce.rollout_micro_batch_size
                gbs=args.global_batch_size, #cfg.model.reinforce.num_rollout_samples
                collate_fn=collate_fn,
                load_gbs=False,
                use_random_sampler=False,
                )
        else:
            valid_dataloader = build_dataloader(
                dataset=valid_ds,
                consumed_samples=args.consumed_valid_samples,
                mbs=1, #cfg.model.reinforce.rollout_micro_batch_size
                gbs=args.global_batch_size, #cfg.model.reinforce.num_rollout_samples
                collate_fn=collate_fn,
                load_gbs=False,
                use_random_sampler=False,
                )
        test_dataloader = valid_dataloader

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        flags = torch.tensor(
            [int(do_train), int(do_valid), int(do_test)],
            dtype=torch.long, device='cuda')
    else:
        flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')

    torch.distributed.broadcast(flags, 0)

    args.do_train = getattr(args, "do_train", False) or flags[0].item()
    args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
    args.do_test = getattr(args, "do_test", False) or flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader

def compute_num_rollout_microbatches(args, dataloader):
    return divide(
        divide(args.global_batch_size, dataloader.batch_sampler.micro_batch_size),
        mpu.get_data_parallel_world_size(),
    )

def batch_pad_to_fixed_len(batch, max_batch_len, pad_token):
    batch_pad = torch.stack(
        [torch.cat([seq, torch.full((max_batch_len - len(seq),), pad_token, dtype=seq.dtype),]) for seq in batch]
    )

    return batch_pad


def collate_with_batch_max_sequence_length(
    data_batch,
    response_token_length,
    eos_id,
    reset_position_ids,
    reset_attention_mask,
    eod_mask_loss,
    generate_masks_and_position_ids,
):
    """collate function that batches by max sequence length
    """
    texts = [item["text"] for item in data_batch]
    loss_multipliers = torch.as_tensor([item["loss_multiplier"] for item in data_batch]).view(len(data_batch), 1)
    lengths = torch.as_tensor([item["length"] for item in data_batch])
    batch_max_length = lengths.max()

    texts = batch_pad_to_fixed_len(texts, response_token_length, eos_id)

    output = {
        "text": texts,
        "length": lengths,
    }

    other = {}
    if generate_masks_and_position_ids:
        # NOTE: the attention mask is 1x1xSxS, which will broadcast on the batch dimension
        attention_masks, loss_masks, position_ids = get_ltor_masks_and_position_ids(
            texts, eos_id, reset_position_ids, reset_attention_mask, eod_mask_loss
        )
        other = {
            "attention_mask": attention_masks,
            # to preserve the loss mask from the dataset
            "loss_mask": loss_masks * loss_multipliers,
            "position_ids": position_ids,
        }

    return output | other

def collate_with_pad_to_max_batch(args, max_seqlen, tokenizer_eos_id, generate_masks_and_position_ids=True):
    """collate function that pads each sequence to the max in the batch"""
    return partial(
        collate_with_batch_max_sequence_length,
        response_token_length=max_seqlen,
        eos_id=tokenizer_eos_id,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        generate_masks_and_position_ids=generate_masks_and_position_ids,
    )
    
@dataclass
class DefaultBatchIterator:
    """The default batch iterator used for getting samples for generation stage 
    """

    sampler_iter: Iterator[int]
    num_microbatches: int
    dataset: Mapping
    collate_fn: Callable

    def __iter__(self):
        for _, ids in zip(range(self.num_microbatches), self.sampler_iter):
            batch = self.collate_fn([self.dataset[index] for index in ids])
            yield batch

class ReinforceRolloutBatch(UserDict):
    @classmethod
    def from_rollout_batches(
        cls: Self, rollout_batches: List[Dict], eos_id: int, rollout_batch_seq_length: Optional[int]
    ) -> Self:
        """Given a list of rollout batches, stack the tensors within and put them in a single dictionary
        """
        stacked_dict = cls()

        for k in sorted(rollout_batches[0]):

            list_of_tensors = [item[k] for item in rollout_batches]

            if all(x.ndim == 1 for x in list_of_tensors):
                tensor = torch.cat(list_of_tensors)
            else:
                pad_value = eos_id if k == "response_tokens" else 0

                list_of_tensors = [row.flatten() for tensor in list_of_tensors for row in tensor]
                # TODO: can we avoid padding locally then padding globally?
                tensor = torch.nn.utils.rnn.pad_sequence(list_of_tensors, batch_first=True, padding_value=pad_value)

                # find the max sequence length globally
                max_seqlen = torch.tensor([tensor.size(-1)], dtype=torch.long, device=torch.cuda.current_device())
                torch.distributed.all_reduce(max_seqlen, op=torch.distributed.ReduceOp.MAX)

                # if rollout_batch_seq_length is None or max_seqlen >= rollout_batch_seq_length:
                #     pad_seq_len = max_seqlen.item()
                # else:
                    # response tokens must be B x S because computing log probs requires us to offset by 1
                pad_seq_len = rollout_batch_seq_length if k == "response_tokens" else rollout_batch_seq_length - 1

                tensor = torch.nn.functional.pad(tensor, (0, pad_seq_len - tensor.size(-1)), value=pad_value)

            stacked_dict[k] = tensor

        return stacked_dict

    def gather_and_balance_globally(self):
        global_rollout_batch = type(self)()

        for k, tensor in self.data.items():
            # with reshard enabled, PP groups turn into DP groups. So need to balance them first and then
            # balance by all the original DP groups
            # NOTE: this logic needs to use the pure parallel state, that is one without sharding but needs
            # to ping the is_trt_llm_reshard variable
            tensor = rebalance_nd_tensor(tensor, group=mpu.get_data_parallel_group())
            global_rollout_batch[k] = tensor

        return global_rollout_batch

    def chunk(self, rank, split_size, seed):
        chunked_rollout_batch = type(self)()

        batch_set = set(tensor.size(0) for tensor in self.data.values())
        assert len(batch_set) == 1, "batch sizes are not the same across the rollout batch"
        B = batch_set.pop()

        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        indices = torch.randperm(B, generator=g_cpu)
        chunks = indices.chunk(split_size)
        my_indices = chunks[rank]

        for k in self.data:
            chunked_rollout_batch[k] = self.data[k][my_indices].clone()

        return chunked_rollout_batch
