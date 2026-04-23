import resource
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import List

from .tool_registry import ToolResult

# Maximum output length to prevent memory issues
MAX_OUTPUT_LEN = 4096


def _sandbox_worker_init(max_memory_mb: int, max_cpu_time_sec: int):
    """Initialize sandbox worker with resource limits."""
    # Memory limit
    mem_bytes = max_memory_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, resource.error):
        pass
    # CPU time limit
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_time_sec, max_cpu_time_sec))
    except (ValueError, resource.error):
        pass


def _execute_code_in_sandbox(code: str) -> dict:
    """Execute Python code in a sandboxed subprocess. Returns dict with success/output/error."""
    import io
    import sys

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_stdout = io.StringIO()
    sys.stderr = captured_stderr = io.StringIO()

    try:
        exec_globals = {"__builtins__": __builtins__}
        exec(code, exec_globals)
        stdout_val = captured_stdout.getvalue()
        stderr_val = captured_stderr.getvalue()
        output = stdout_val
        if stderr_val:
            output += "\n[stderr]\n" + stderr_val
        # Truncate
        if len(output) > MAX_OUTPUT_LEN:
            output = output[:MAX_OUTPUT_LEN] + "\n... [output truncated]"
        return {"success": True, "output": output, "error": None}
    except Exception:
        tb = traceback.format_exc()
        if len(tb) > MAX_OUTPUT_LEN:
            tb = tb[:MAX_OUTPUT_LEN] + "\n... [traceback truncated]"
        return {"success": False, "output": "", "error": tb}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@dataclass
class SandboxConfig:
    pool_size: int = 8
    max_memory_mb: int = 512
    max_wall_time_sec: float = 30.0
    max_cpu_time_sec: int = 30


class SandboxPool:
    """Process pool for sandboxed code execution."""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self._pool = ProcessPoolExecutor(
            max_workers=config.pool_size,
            initializer=_sandbox_worker_init,
            initargs=(config.max_memory_mb, config.max_cpu_time_sec),
        )

    def execute_code(self, code: str) -> ToolResult:
        """Execute a single code snippet with timeout."""
        try:
            future = self._pool.submit(_execute_code_in_sandbox, code)
            result = future.result(timeout=self.config.max_wall_time_sec)
            return ToolResult(
                success=result["success"],
                output=result["output"],
                error=result["error"],
            )
        except FuturesTimeoutError:
            return ToolResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.config.max_wall_time_sec}s",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def execute_batch(self, codes: List[str]) -> List[ToolResult]:
        """Execute a batch of code snippets in parallel."""
        futures = [self._pool.submit(_execute_code_in_sandbox, code) for code in codes]
        results = []
        for future in futures:
            try:
                result = future.result(timeout=self.config.max_wall_time_sec)
                results.append(ToolResult(
                    success=result["success"],
                    output=result["output"],
                    error=result["error"],
                ))
            except FuturesTimeoutError:
                results.append(ToolResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {self.config.max_wall_time_sec}s",
                ))
            except Exception as e:
                results.append(ToolResult(success=False, output="", error=str(e)))
        return results

    def shutdown(self):
        self._pool.shutdown(wait=False)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
