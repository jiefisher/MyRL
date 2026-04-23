from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str] = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    executor: Optional[Callable] = None
    timeout: float = 30.0
    requires_sandbox: bool = False


class ToolRegistry:
    """Manages tool registration and execution dispatch."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_definitions(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def register_builtin_tools(self):
        """Register built-in tools like python_execute."""
        self.register(ToolDefinition(
            name="python_execute",
            description="Execute Python code and return the result.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
            executor=None,  # Routed to sandbox pool
            timeout=30.0,
            requires_sandbox=True,
        ))

    def execute(self, name: str, arguments: Dict[str, Any], sandbox_pool=None) -> ToolResult:
        """Execute a tool by name. Sandbox tools are routed to the sandbox pool."""
        tool = self.get(name)
        if tool is None:
            return ToolResult(success=False, output="", error=f"Unknown tool: {name}")

        if tool.requires_sandbox:
            if sandbox_pool is None:
                return ToolResult(success=False, output="", error="Sandbox pool not available")
            if name == "python_execute":
                code = arguments.get("code", "")
                return sandbox_pool.execute_code(code)
            return ToolResult(success=False, output="", error=f"No sandbox handler for tool: {name}")

        if tool.executor is not None:
            try:
                result = tool.executor(arguments)
                if isinstance(result, ToolResult):
                    return result
                return ToolResult(success=True, output=str(result))
            except Exception as e:
                return ToolResult(success=False, output="", error=str(e))

        return ToolResult(success=False, output="", error=f"No executor for tool: {name}")
