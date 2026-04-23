from dataclasses import dataclass
from typing import List

import torch

from .trajectory import Trajectory, TurnRecord


class AgentRewardComputer:
    """Computes final + process rewards for agent trajectories."""

    def __init__(
        self,
        final_reward_weight: float = 0.8,
        process_reward_weight: float = 0.2,
        tool_success_reward: float = 0.1,
        tool_failure_penalty: float = -0.1,
        turn_penalty: float = -0.05,
        format_correct_reward: float = 0.05,
    ):
        self.final_reward_weight = final_reward_weight
        self.process_reward_weight = process_reward_weight
        self.tool_success_reward = tool_success_reward
        self.tool_failure_penalty = tool_failure_penalty
        self.turn_penalty = turn_penalty
        self.format_correct_reward = format_correct_reward

    def compute_turn_reward(self, turn: TurnRecord) -> float:
        """Compute process reward for a single turn."""
        reward = 0.0
        if turn.tool_call is not None:
            # Format correct reward
            reward += self.format_correct_reward
            # Tool execution result
            if turn.tool_result is not None:
                if turn.tool_result.success:
                    reward += self.tool_success_reward
                else:
                    reward += self.tool_failure_penalty
        # Per-turn penalty to encourage efficiency
        if not turn.is_final:
            reward += self.turn_penalty
        return reward

    def compute_trajectory_reward(
        self,
        trajectory: Trajectory,
        final_reward: float,
    ) -> dict:
        """Compute combined reward for a trajectory.

        Args:
            trajectory: The multi-turn trajectory.
            final_reward: The final reward from the reward function (e.g., rpf.compute_score).

        Returns:
            dict with:
                total_reward: weighted combination of final and process rewards
                final_reward: the final reward
                process_reward: sum of per-turn process rewards
                per_turn_rewards: list of per-turn process rewards
        """
        per_turn_rewards = []
        for turn in trajectory.turns:
            per_turn_rewards.append(self.compute_turn_reward(turn))

        process_reward = sum(per_turn_rewards)
        total_reward = (
            self.final_reward_weight * final_reward
            + self.process_reward_weight * process_reward
        )

        return {
            "total_reward": total_reward,
            "final_reward": final_reward,
            "process_reward": process_reward,
            "per_turn_rewards": per_turn_rewards,
        }

    def assign_process_rewards_to_tokens(
        self,
        trajectory: Trajectory,
        per_turn_rewards: List[float],
        process_rewards_tensor: torch.FloatTensor,
        prompt_length: int,
    ) -> torch.FloatTensor:
        """Distribute per-turn process rewards uniformly to generated tokens of each turn.

        Modifies process_rewards_tensor in-place and returns it.
        """
        pos = prompt_length
        for turn, turn_reward in zip(trajectory.turns, per_turn_rewards):
            gen_len = len(turn.generated_token_ids)
            if gen_len > 0:
                per_token_reward = turn_reward / gen_len
                end_pos = min(pos + gen_len, process_rewards_tensor.size(0))
                process_rewards_tensor[pos:end_pos] = per_token_reward
            pos += gen_len
            # Skip tool result tokens
            pos += len(turn.tool_result_token_ids)
        return process_rewards_tensor
