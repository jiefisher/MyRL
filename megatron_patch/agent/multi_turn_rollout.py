from typing import List, Optional

from .tool_registry import ToolRegistry, ToolResult
from .tool_parser import ToolCallParser
from .tool_formatter import ToolFormatter
from .sandbox_pool import SandboxPool
from .trajectory import Trajectory, TurnRecord


class MultiTurnRolloutOrchestrator:
    """Orchestrates multi-turn rollout: generate -> parse -> execute tool -> append -> repeat."""

    def __init__(
        self,
        inference_engine,
        sampling_params,
        tokenizer,
        tool_registry: ToolRegistry,
        tool_parser: ToolCallParser,
        tool_formatter: ToolFormatter,
        sandbox_pool: Optional[SandboxPool],
        max_turns: int = 5,
        max_total_tokens: int = 16384,
        bio_tool_executor=None,
    ):
        self.inference_engine = inference_engine
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        self.tool_registry = tool_registry
        self.tool_parser = tool_parser
        self.tool_formatter = tool_formatter
        self.sandbox_pool = sandbox_pool
        self.max_turns = max_turns
        self.max_total_tokens = max_total_tokens
        self.bio_tool_executor = bio_tool_executor

    def _execute_tool(self, global_idx: int, parsed_call, turn: TurnRecord):
        """Execute a single tool call and populate turn with results."""
        tool_name = parsed_call.tool_name

        # Route bio workflow tools to BioToolExecutor
        if self.bio_tool_executor is not None and tool_name.startswith("workflow_"):
            result = self.bio_tool_executor.execute(
                global_idx, tool_name, parsed_call.arguments,
            )
        elif tool_name == "python_execute" and self.sandbox_pool is not None:
            code = parsed_call.arguments.get("code", "")
            result = self.sandbox_pool.execute_code(code)
        elif self.sandbox_pool is not None:
            result = self.tool_registry.execute(
                tool_name, parsed_call.arguments, self.sandbox_pool,
            )
        else:
            result = ToolResult(success=False, output="", error=f"No executor for tool: {tool_name}")

        turn.tool_result = result
        formatted = self.tool_formatter.format_tool_result(parsed_call, result)
        turn.tool_result_text = formatted
        turn.tool_result_token_ids = self.tokenizer.encode(formatted, add_special_tokens=False)

    def rollout_batch(
        self,
        prompt_token_ids_list: List[List[int]],
        ground_truths: List[str],
    ) -> List[Trajectory]:
        """Run multi-turn rollout for a batch of prompts.

        Args:
            prompt_token_ids_list: List of tokenized prompts.
            ground_truths: Corresponding ground truth labels.

        Returns:
            List of Trajectory objects, one per sample.
        """
        batch_size = len(prompt_token_ids_list)
        trajectories = [
            Trajectory(
                original_prompt_tokens=list(prompt_token_ids_list[i]),
                ground_truth=ground_truths[i],
            )
            for i in range(batch_size)
        ]

        # Reset bio envs for this batch
        if self.bio_tool_executor is not None:
            self.bio_tool_executor.reset_all()

        # Track which samples are still active (not yet produced final answer)
        active_indices = list(range(batch_size))
        # Current token sequences for each active sample
        current_tokens = [list(prompt_token_ids_list[i]) for i in range(batch_size)]

        for turn_idx in range(self.max_turns):
            if not active_indices:
                break

            # 1. Batch vLLM generation for all active samples
            active_prompts = [current_tokens[i] for i in active_indices]
            outputs = self.inference_engine.generate(
                prompt_token_ids=active_prompts,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            # 2. Parse each output
            needs_tool = []  # (global_idx, parsed_tool_call, turn)
            for local_idx, (global_idx, output) in enumerate(zip(active_indices, outputs)):
                gen_token_ids = list(output.outputs[0].token_ids)
                gen_text = output.outputs[0].text

                # Extract logprobs
                logprobs_list = []
                if output.outputs[0].logprobs:
                    for token_id, logprob_obj in zip(
                        output.outputs[0].token_ids,
                        output.outputs[0].logprobs,
                    ):
                        if logprob_obj:
                            for token, logprob in logprob_obj.items():
                                logprobs_list.append(logprob.logprob)
                                break  # only first (top) logprob
                        else:
                            logprobs_list.append(0.0)

                # Check if this is a final answer or contains a tool call
                parsed_call = self.tool_parser.parse(gen_text)

                is_final = parsed_call is None
                # Also check if EOS was generated
                if gen_token_ids and gen_token_ids[-1] == self.tokenizer.eos_token_id:
                    is_final = True
                # workflow_finalize means this is the last turn
                if parsed_call is not None and parsed_call.tool_name == "workflow_finalize":
                    is_final = True

                turn = TurnRecord(
                    turn_idx=turn_idx,
                    generated_token_ids=gen_token_ids,
                    generated_text=gen_text,
                    generation_logprobs=logprobs_list,
                    tool_call=parsed_call,
                    is_final=is_final,
                )
                trajectories[global_idx].turns.append(turn)

                if parsed_call is not None:
                    needs_tool.append((global_idx, parsed_call, turn))

            # 3. Execute tools
            # Batch sandbox calls separately for efficiency
            sandbox_batch = []
            other_calls = []
            for global_idx, parsed_call, turn in needs_tool:
                if parsed_call.tool_name == "python_execute" and self.sandbox_pool is not None:
                    sandbox_batch.append((global_idx, parsed_call, turn))
                else:
                    other_calls.append((global_idx, parsed_call, turn))

            # Execute non-sandbox tools (bio tools, etc.)
            for global_idx, parsed_call, turn in other_calls:
                self._execute_tool(global_idx, parsed_call, turn)

            # Batch sandbox execution
            if sandbox_batch:
                codes = [pc.arguments.get("code", "") for _, pc, _ in sandbox_batch]
                results = self.sandbox_pool.execute_batch(codes)
                for (global_idx, parsed_call, turn), result in zip(sandbox_batch, results):
                    turn.tool_result = result
                    formatted = self.tool_formatter.format_tool_result(parsed_call, result)
                    turn.tool_result_text = formatted
                    turn.tool_result_token_ids = self.tokenizer.encode(
                        formatted, add_special_tokens=False
                    )

            # 4. Update current_tokens and active_indices for next turn
            new_active = []
            for global_idx, parsed_call, turn in needs_tool:
                # Append generated tokens + tool result tokens to current sequence
                current_tokens[global_idx].extend(turn.generated_token_ids)
                current_tokens[global_idx].extend(turn.tool_result_token_ids)

                # Check total token budget
                if len(current_tokens[global_idx]) >= self.max_total_tokens:
                    turn.is_final = True
                elif not turn.is_final:
                    new_active.append(global_idx)

            active_indices = new_active

        return trajectories
