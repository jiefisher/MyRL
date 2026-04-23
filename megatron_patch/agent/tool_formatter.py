from typing import List, Optional

from .tool_registry import ToolResult
from .tool_parser import ParsedToolCall


class ToolFormatter:
    """Formats tool results as conversation turns and builds multi-turn prompts."""

    def __init__(self, tokenizer, format: str = "qwen3"):
        self.tokenizer = tokenizer
        self.format = format

    def format_tool_result(self, tool_call: ParsedToolCall, result: ToolResult) -> str:
        """Format a tool execution result as a conversation turn string."""
        if self.format == "qwen3":
            return self._format_qwen3(tool_call, result)
        elif self.format == "react":
            return self._format_react(tool_call, result)
        elif self.format == "function_call":
            return self._format_function_call(tool_call, result)
        raise ValueError(f"Unknown format: {self.format}")

    def _format_qwen3(self, tool_call: ParsedToolCall, result: ToolResult) -> str:
        output = result.output if result.success else f"Error: {result.error}"
        # Truncate long outputs
        if len(output) > 4096:
            output = output[:4096] + "\n... [output truncated]"
        return f"<tool_response>\n{output}\n</tool_response>"

    def _format_react(self, tool_call: ParsedToolCall, result: ToolResult) -> str:
        output = result.output if result.success else f"Error: {result.error}"
        if len(output) > 4096:
            output = output[:4096] + "\n... [output truncated]"
        return f"Observation: {output}"

    def _format_function_call(self, tool_call: ParsedToolCall, result: ToolResult) -> str:
        output = result.output if result.success else f"Error: {result.error}"
        if len(output) > 4096:
            output = output[:4096] + "\n... [output truncated]"
        return f"```function_result\n{output}\n```"

    def build_multi_turn_prompt(
        self,
        original_prompt_tokens: List[int],
        turn_texts: List[str],
    ) -> List[int]:
        """Build a multi-turn prompt by appending turn texts to the original prompt tokens.

        Args:
            original_prompt_tokens: The initial prompt token IDs.
            turn_texts: List of text strings (alternating assistant generation + tool result).

        Returns:
            Combined token IDs for the full multi-turn conversation.
        """
        all_tokens = list(original_prompt_tokens)
        for text in turn_texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
        return all_tokens

    def build_continuation_tokens(
        self,
        original_prompt_tokens: List[int],
        assistant_texts: List[str],
        tool_result_texts: List[str],
    ) -> List[int]:
        """Build tokens for continuing generation after tool results.

        Interleaves assistant generations and tool results, then returns
        the full token sequence to use as the next prompt.
        """
        all_tokens = list(original_prompt_tokens)
        for i in range(len(assistant_texts)):
            # Encode assistant generation
            asst_tokens = self.tokenizer.encode(assistant_texts[i], add_special_tokens=False)
            all_tokens.extend(asst_tokens)
            # Encode tool result if available
            if i < len(tool_result_texts):
                result_tokens = self.tokenizer.encode(tool_result_texts[i], add_special_tokens=False)
                all_tokens.extend(result_tokens)
        return all_tokens
