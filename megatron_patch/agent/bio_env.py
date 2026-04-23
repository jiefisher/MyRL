"""BioWorkflow环境封装 — 将bioworkflow_gym包装为多轮交互工具。

每个sample维护独立的BioWorkflowEnv实例，通过BioToolExecutor统一管理。
"""

from __future__ import annotations

import sys
import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .tool_registry import ToolDefinition, ToolResult, ToolRegistry

# 动态添加bioworkflow_gym路径
_BIO_GYM_PATH = os.path.dirname(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
))

def _ensure_bio_gym_importable(bio_gym_root: str = "/Users/fangjie/Desktop/bio"):
    """确保bioworkflow_gym可以被import。"""
    if bio_gym_root not in sys.path:
        sys.path.insert(0, bio_gym_root)


class BioWorkflowEnv:
    """单个sample的有状态workflow构建环境。"""

    def __init__(self, bio_gym_root: str = "/Users/fangjie/Desktop/bio"):
        _ensure_bio_gym_importable(bio_gym_root)

        from bioworkflow_gym.models import VerifierScores
        from bioworkflow_gym.parser import WorkflowParser
        from bioworkflow_gym.dag import DAGValidator
        from bioworkflow_gym.param_validator import ParamValidator
        from bioworkflow_gym.stub_registry import StubRegistry
        from bioworkflow_gym.stub_executor import StubExecutor
        from bioworkflow_gym.schema_validator import SchemaValidator
        from bioworkflow_gym.reward import RewardCalculator
        from bioworkflow_gym.domain.rnaseq import (
            RNASEQ_CHUNKS, RNASEQ_REQUIRED_STAGES, register_rnaseq,
            check_scientific_rules, RNASEQ_PROCESSES,
        )

        self._stub_registry = StubRegistry()
        self._domain = register_rnaseq(self._stub_registry)
        self._parser = WorkflowParser(self._domain["processes"])
        self._dag_validator = DAGValidator(
            required_stages=RNASEQ_REQUIRED_STAGES,
            chunks=RNASEQ_CHUNKS,
        )
        self._param_validator = ParamValidator(self._domain["param_constraints"])
        self._stub_executor = StubExecutor(self._stub_registry)
        self._schema_validator = SchemaValidator()
        self._reward_calc = RewardCalculator(chunks=RNASEQ_CHUNKS)
        self._check_scientific_rules = check_scientific_rules
        self._VerifierScores = VerifierScores
        self._available_processes = RNASEQ_PROCESSES

        # 可变状态：逐步构建的workflow
        self._workflow_state: Dict[str, Any] = {
            "name": "agent_workflow",
            "processes": [],
            "connections": [],
            "params": {},
        }
        self._step_count = 0
        self._finalized = False

    def add_process(self, process_name: str) -> ToolResult:
        """添加一个process到workflow。"""
        if self._finalized:
            return ToolResult(False, "", "Workflow already finalized.")

        if process_name not in self._available_processes:
            available = list(self._available_processes.keys())
            return ToolResult(
                False, "",
                f"Unknown process '{process_name}'. Available: {available}",
            )

        if process_name in self._workflow_state["processes"]:
            return ToolResult(False, "", f"Process '{process_name}' already added.")

        self._workflow_state["processes"].append(process_name)
        self._step_count += 1
        n = len(self._workflow_state["processes"])
        total = len(self._available_processes)
        return ToolResult(
            True,
            f"Process {process_name} added. {n}/{total} processes defined. "
            f"Current workflow: {self._workflow_state['processes']}",
        )

    def connect(self, src: str, dst: str) -> ToolResult:
        """添加一条连接 (格式: "PROCESS.port")。"""
        if self._finalized:
            return ToolResult(False, "", "Workflow already finalized.")

        src_parts = src.split(".")
        dst_parts = dst.split(".")
        if len(src_parts) != 2 or len(dst_parts) != 2:
            return ToolResult(
                False, "",
                f"Invalid format. Expected 'PROCESS.port', got '{src}' -> '{dst}'",
            )

        src_proc, src_port = src_parts
        dst_proc, dst_port = dst_parts

        # 检查process是否已添加
        for proc_name in [src_proc, dst_proc]:
            if proc_name not in self._workflow_state["processes"]:
                return ToolResult(
                    False, "",
                    f"Process '{proc_name}' not in workflow. Add it first.",
                )

        self._workflow_state["connections"].append({"src": src, "dst": dst})
        self._step_count += 1
        n = len(self._workflow_state["connections"])
        return ToolResult(
            True,
            f"Connection {src} -> {dst} added. {n} connections total.",
        )

    def set_param(self, process: str, key: str, value: Any) -> ToolResult:
        """设置process参数。"""
        if self._finalized:
            return ToolResult(False, "", "Workflow already finalized.")

        if process not in self._workflow_state["processes"]:
            return ToolResult(
                False, "",
                f"Process '{process}' not in workflow. Add it first.",
            )

        if process not in self._workflow_state["params"]:
            self._workflow_state["params"][process] = {}
        self._workflow_state["params"][process][key] = value
        self._step_count += 1
        return ToolResult(
            True,
            f"Parameter {process}.{key} = {value} set. "
            f"Current params for {process}: {self._workflow_state['params'][process]}",
        )

    def validate(self) -> ToolResult:
        """运行部分验证（Layer 1），返回中间反馈。"""
        workflow, syntax_score, syntax_errors = self._parser.parse(self._workflow_state)
        self._step_count += 1

        if workflow is None:
            return ToolResult(
                True,
                f"Validation: syntax={syntax_score:.2f}. "
                f"Errors: {syntax_errors}. Fix these before finalizing.",
            )

        dag_score, dag, dag_errors = self._dag_validator.validate(workflow)
        param_score, param_errors = self._param_validator.validate(workflow)

        all_errors = syntax_errors + dag_errors + param_errors
        msg = (
            f"Validation scores: syntax={syntax_score:.2f}, dag={dag_score:.2f}, "
            f"param={param_score:.2f}."
        )
        if all_errors:
            msg += f" Errors: {all_errors[:5]}"
            if len(all_errors) > 5:
                msg += f" ... and {len(all_errors) - 5} more"
        else:
            msg += " No errors found. Ready to finalize."
        return ToolResult(True, msg)

    def finalize(self) -> Tuple[ToolResult, float, Dict[str, Any]]:
        """运行完整6层验证，返回最终结果和reward。"""
        self._finalized = True
        self._step_count += 1

        # Layer 1: Syntax
        workflow, syntax_score, syntax_errors = self._parser.parse(self._workflow_state)
        if workflow is None:
            scores = self._VerifierScores(syntax=syntax_score)
            result = self._reward_calc.compute(scores)
            return (
                ToolResult(True, f"Finalized. Syntax failed. Reward: {result.total_reward:.4f}"),
                result.total_reward,
                {"scores": scores, "errors": syntax_errors},
            )

        # Layer 1: DAG
        dag_score, dag, dag_errors = self._dag_validator.validate(workflow)

        # Layer 1: Param
        param_score, param_errors = self._param_validator.validate(workflow)

        # Layer 2: Stub execution
        stub_score, stub_results, stub_errors = 0.0, [], []
        if dag is not None:
            stub_score, stub_results, stub_errors = self._stub_executor.execute(workflow, dag)

        # Layer 2: Schema
        schema_score, schema_errors = self._schema_validator.validate(
            stub_results, self._stub_registry
        )

        # Scientific rules
        proc_order = [p.name for p in workflow.processes]
        sci_score, sci_errors = self._check_scientific_rules(proc_order, workflow.params)

        scores = self._VerifierScores(
            syntax=syntax_score,
            dag=dag_score,
            param=param_score,
            stub_run=stub_score,
            output_schema=schema_score,
            scientific=sci_score,
        )
        result = self._reward_calc.compute(scores)

        all_errors = syntax_errors + dag_errors + param_errors + stub_errors + schema_errors + sci_errors
        msg = (
            f"Finalized. Reward: {result.total_reward:.4f}. "
            f"Scores: syntax={syntax_score:.2f}, dag={dag_score:.2f}, "
            f"param={param_score:.2f}, stub={stub_score:.2f}, "
            f"schema={schema_score:.2f}, scientific={sci_score:.2f}. "
            f"All pass: {result.details.get('all_pass', False)}"
        )
        return (
            ToolResult(True, msg),
            result.total_reward,
            {"scores": scores, "errors": all_errors, "gates": result.gates},
        )

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def workflow_state(self) -> Dict[str, Any]:
        return self._workflow_state


class BioToolExecutor:
    """管理多个sample的BioWorkflowEnv实例，提供批量工具执行。"""

    def __init__(self, bio_gym_root: str = "/Users/fangjie/Desktop/bio"):
        self.bio_gym_root = bio_gym_root
        self._envs: Dict[int, BioWorkflowEnv] = {}

    def get_or_create_env(self, sample_id: int) -> BioWorkflowEnv:
        if sample_id not in self._envs:
            self._envs[sample_id] = BioWorkflowEnv(self.bio_gym_root)
        return self._envs[sample_id]

    def reset(self, sample_id: int):
        """重置某个sample的环境。"""
        self._envs[sample_id] = BioWorkflowEnv(self.bio_gym_root)

    def reset_all(self):
        """重置所有环境。"""
        self._envs.clear()

    def execute(self, sample_id: int, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """执行一个工具调用。"""
        env = self.get_or_create_env(sample_id)

        if tool_name == "workflow_add_process":
            return env.add_process(arguments.get("process_name", ""))
        elif tool_name == "workflow_connect":
            return env.connect(arguments.get("src", ""), arguments.get("dst", ""))
        elif tool_name == "workflow_set_param":
            return env.set_param(
                arguments.get("process", ""),
                arguments.get("key", ""),
                arguments.get("value"),
            )
        elif tool_name == "workflow_validate":
            return env.validate()
        elif tool_name == "workflow_finalize":
            result, reward, info = env.finalize()
            return result
        else:
            return ToolResult(False, "", f"Unknown bio tool: {tool_name}")

    def finalize(self, sample_id: int) -> Tuple[ToolResult, float, Dict[str, Any]]:
        """Finalize并返回reward（供reward计算使用）。"""
        env = self.get_or_create_env(sample_id)
        if env._finalized:
            # 已经finalize过，重新运行获取reward
            env._finalized = False
        return env.finalize()

    def execute_batch(
        self,
        sample_ids: List[int],
        tool_names: List[str],
        arguments_list: List[Dict[str, Any]],
    ) -> List[ToolResult]:
        """批量执行工具调用。"""
        return [
            self.execute(sid, name, args)
            for sid, name, args in zip(sample_ids, tool_names, arguments_list)
        ]


def register_bio_tools(registry: ToolRegistry):
    """注册bioworkflow相关的工具到ToolRegistry。"""
    registry.register(ToolDefinition(
        name="workflow_add_process",
        description="Add a bioinformatics process to the workflow. "
                    "Available: FASTQC, FASTP, STAR_ALIGN, SAMTOOLS_SORT, FEATURECOUNTS, DESEQ2, MULTIQC",
        parameters_schema={
            "type": "object",
            "properties": {
                "process_name": {"type": "string", "description": "Name of the process to add"},
            },
            "required": ["process_name"],
        },
        requires_sandbox=False,
    ))
    registry.register(ToolDefinition(
        name="workflow_connect",
        description="Connect an output port of one process to an input port of another. "
                    "Format: 'PROCESS.port'",
        parameters_schema={
            "type": "object",
            "properties": {
                "src": {"type": "string", "description": "Source in PROCESS.port format"},
                "dst": {"type": "string", "description": "Destination in PROCESS.port format"},
            },
            "required": ["src", "dst"],
        },
        requires_sandbox=False,
    ))
    registry.register(ToolDefinition(
        name="workflow_set_param",
        description="Set a parameter for a process in the workflow.",
        parameters_schema={
            "type": "object",
            "properties": {
                "process": {"type": "string", "description": "Process name"},
                "key": {"type": "string", "description": "Parameter name"},
                "value": {"description": "Parameter value"},
            },
            "required": ["process", "key", "value"],
        },
        requires_sandbox=False,
    ))
    registry.register(ToolDefinition(
        name="workflow_validate",
        description="Run partial validation (syntax, DAG, parameters) on the current workflow state. "
                    "Use this to check for errors before finalizing.",
        parameters_schema={"type": "object", "properties": {}},
        requires_sandbox=False,
    ))
    registry.register(ToolDefinition(
        name="workflow_finalize",
        description="Finalize the workflow and run full 6-layer verification "
                    "(syntax, DAG, params, stub execution, schema, scientific rules). "
                    "Call this when the workflow is complete.",
        parameters_schema={"type": "object", "properties": {}},
        requires_sandbox=False,
    ))
