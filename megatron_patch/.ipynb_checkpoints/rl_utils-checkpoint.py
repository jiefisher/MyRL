# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""helper functions for PPO training"""

import operator

import torch
from typing import Optional
from torch.masked import as_masked_tensor
from megatron.training import print_rank_0

def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
    global_normalization_factor: Optional[torch.Tensor | float] = None,
):
    """Computes the mean of a microbatch, using a global statistic as the normalization factor."""
    normalization_factor = (
        torch.sum(mask, dim=dim)
        if global_normalization_factor is None
        else global_normalization_factor
    )
    return torch.sum(values * mask, dim=dim) / (normalization_factor + 1e-8)

def calculate_kl_penalty_joschu2020(
    logprobs_policy: torch.Tensor, logprobs_reference: torch.Tensor
) -> torch.Tensor:
    """Calculates a per-token estimate of the KL Divergence between two log_probs.

    From Schulman 2020, always positive.

    logprobs_policy:    torch.Tensor (b, s)
    logprobs_reference: torch.Tensor (b, s)
    """
    r = logprobs_reference - logprobs_policy
    return torch.exp(r) - r - 1

def calculate_baseline_and_std_per_prompt(
    prompts: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    leave_one_out_baseline: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to compute a baseline for each (prompt, response) pair in the batch.

    The same baseline is calculated for each prompt. Samples set to 0 in 'valid_mask'
    are not included in the baseline calculation.

    prompts:    tensor (b, s)     Tensor of prompts the model used. May be on any device
    rewards:    tensor (b,)       Float-valued rewards. May be on any device
    valid_mask: tensor (b,)       Vector of 0/1, where 0 is to ignore and 1 is to keep
    leave_one_out_baseline: bool  Compute an unbiased baseline by leaving out the sample that
                                  the baseline is for (from RLOO https://arxiv.org/abs/2402.14740)

    Returns:
    tensor (b,), tensor (b,) of baselines and std on the same device as 'rewards'
    """
    unique_prompts = torch.unique(prompts, dim=0)
    # print_rank_0(unique_prompts.shape)
    baseline = torch.zeros_like(rewards)
    sq_baseline = torch.zeros_like(rewards)
    device_ordinal = rewards.get_device()
    if device_ordinal == -1:
        reward_device = torch.device("cpu")
    else:
        reward_device = rewards.get_device()

    for i in range(len(unique_prompts)):
        is_matching_prompt = (prompts == unique_prompts[i]).all(1)
        prompt_idx = torch.arange(len(prompts), device=reward_device)[
            is_matching_prompt
        ]

        if leave_one_out_baseline:
            baseline_mask_matrix = (1 - torch.eye(len(prompt_idx))).to(reward_device)
        else:
            baseline_mask_matrix = torch.ones((len(prompt_idx), len(prompt_idx))).to(
                reward_device
            )
        if valid_mask[prompt_idx].sum() <= 1:
            # Ignore sample: there are no valid responses, so set baseline equal to reward
            # to ignore it in the loss computation
            baseline[prompt_idx] = rewards[prompt_idx]
        else:
            num_valid = valid_mask[prompt_idx].float().sum() - int(
                leave_one_out_baseline
            )
            prompt_baseline = (
                torch.matmul(
                    baseline_mask_matrix, rewards[prompt_idx] * valid_mask[prompt_idx]
                )
                / num_valid
            )
            prompt_baseline_square = (
                torch.matmul(
                    baseline_mask_matrix,
                    (rewards[prompt_idx] ** 2) * valid_mask[prompt_idx],
                )
                / num_valid
            )

            baseline[prompt_idx] = prompt_baseline
            sq_baseline[prompt_idx] = prompt_baseline_square

    std = (sq_baseline - baseline.square()).sqrt().nan_to_num(0)
    return baseline, std
    
def calculate_advantages_and_returns(values, rewards, discount_factor, gae_lambda, mask=None):
    """calculate the per token advantages and returns for the entire sequence

    Args:
        values, rewards (torch.Tensor): shape of B x (S-1)
    """
    if mask is not None:
        # need the masking here because our sentence might not span the entire sequence length
        values = values * mask
        rewards = rewards * mask

    last_gae_lam = 0
    advantages = torch.zeros_like(rewards)
    max_seq_len = values.size(-1)

    for i in reversed(range(max_seq_len)):
        if i == max_seq_len - 1:
            next_values = 0.0  # Last element has next_value==0.0
        else:
            next_values = values[:, i + 1]  # Get value from next position.
        delta = rewards[:, i] + discount_factor * next_values - values[:, i]
        last_gae_lam = delta + discount_factor * gae_lambda * last_gae_lam
        advantages[:, i] = last_gae_lam

    returns = advantages + values
    return advantages, returns


def calculate_entropy(log_probs, mask=None):
    """calculate the entropy, with an optional mask

    Args:
        log_probs (torch.Tensor): Tensor of log probs with shape [B x S x V]
        mask (torch.Tensor): Tensor of masks on the sequence length with shape B x S
    """
    entropy_unmasked = -torch.sum(log_probs.exp() * log_probs, dim=-1)
    return entropy_unmasked.mean() if mask is None else masked_mean(entropy_unmasked, mask)


def calculate_ppo_rewards(values, rewards, response_lengths, init_policy_kl, penalty_factor=0.0):
    """the reward should be defined on the last valid action"""

    rewards_sequence = torch.zeros_like(values)

    idx = (response_lengths - 2).clamp(min=0, max=None)

    rewards_sequence[torch.arange(rewards_sequence.size(0)), idx] = rewards.flatten()

    return rewards_sequence - penalty_factor * init_policy_kl


def calculate_kl_penalty(log_probs_a, log_probs_b, use_absolute_kl=True):
    """Calculates a per-token estimate of the KL Divergence between two log_probs.
    """
    init_policy_kl = log_probs_a - log_probs_b
    if use_absolute_kl:
        init_policy_kl = init_policy_kl.abs()

    return init_policy_kl


def create_mask(values, prompt_lengths, response_lengths):
    """Creates a mask to only keep the values in the sequence that are between prompt_lengths and sentence_lengths.
    This results in removing the prompt tokens, and removing the padding at the end of the sequence.
    """
    mask = torch.zeros_like(values)
    for i in range(mask.size(0)):
        # Do prompt_length - 1 to remove the first log prob. But keep sentence_length
        # as it is because we want to include one EOS token.
        # print(prompt_lengths[i],response_lengths[i])
        mask[i, prompt_lengths[i] - 1 : min(response_lengths[i] - 1,mask.shape[1])] = 1.0
    return mask


def select_topk(batch, num_select=1):
    """
    Function to select the topk responses for each unique prompt in a batch. 
    Please note that this function samples the same top response for each identical prompt.
    Duplicate prompts in the same batch may cause unexpected behavior.
    """
    unique_prompts = torch.unique(batch["prompt_tokens"], dim=0)
    selected_idx = []

    for i in range(len(unique_prompts)):
        is_matching_prompt = (batch["prompt_tokens"] == unique_prompts[i]).all(1)
        prompt_idx = torch.arange(len(batch["prompt_tokens"]))[is_matching_prompt]
        sorted_idx = zip(prompt_idx, batch["rewards"][is_matching_prompt])
        sorted_idx = sorted(sorted_idx, key=operator.itemgetter(1))
        selected_idx += [x[0].item() for x in sorted_idx[-1 * num_select :]]

    selected_batch = {k: batch[k][selected_idx] for k in batch.keys()}
    return selected_batch


def calculate_rloo_baseline(prompts, reward, k):
    """
    Function to select the RLOO baseline for each (prompt, response) pair in the batch. 
    The same baseline is calculated for each prompt. Masked samples are not included
    in the baseline calculation.
    """
    unique_prompts = torch.unique(prompts, dim=0)
    baseline = torch.zeros_like(reward)
    reward_device = reward.get_device()
    
    if reward_device == -1:
        reward_device = "cpu"
    mask = torch.ones(reward.shape[0], device=reward_device)
    # reward = reward.reshape(rloo_k, local_batch_size)
    # print("reward",reward.shape)
    # print(reward)
    for i in range(len(unique_prompts)):
        is_matching_prompt = (prompts == unique_prompts[i]).all(1)
        prompt_idx = torch.arange(len(prompts), device=reward_device)[is_matching_prompt]
        rloo_mat = (1 - torch.eye(len(prompt_idx))).to(reward_device)

        # if mask[prompt_idx].sum() <= 1:
        #     # Ignore sample: set baseline equal to reward
        #     baseline[prompt_idx] = reward[prompt_idx]
        # else:
        rloo = torch.matmul(rloo_mat, reward[prompt_idx] * mask[prompt_idx]) / (mask[prompt_idx].sum() - 1)
        baseline[prompt_idx] = rloo

    return baseline
