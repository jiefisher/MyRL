<div align="center">
👋 MyRL — A Lightweight Reinforcement Learning Training Framework
<br>
<br>
</div>

MyRL is an open-source, high-performance, lightweight RLHF / RL training framework built on **Megatron-LM** + **vLLM**. It targets reinforcement learning for large language models, with native support for 3D parallelism (TP / PP / CP) and MoE expert parallelism (ETP / EP), and uses vLLM to accelerate the rollout-phase generation. It also ships with multi-turn agent rollout, a tool-call sandbox, and process-reward support, making it easy to build agentic RL training pipelines.

## Table of Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Training Workflow](#training-workflow)
- [Design Philosophy](#design-philosophy)
- [Quick Start](#quick-start)
- [Parameters](#parameters)
- [Benchmarks](#benchmarks)
- [Roadmap](#roadmap)
- [References](#references)

## Key Features

- **Training backend**: Megatron-LM, native TP / PP / CP / ETP / EP, covers both Dense and MoE models
- **Inference backend**: vLLM with sleep / wake-up mode — exclusive GPU during rollout, released during training
- **Algorithm**: GRPO with KL penalty, PPO-style clipping, token-level Importance Sampling, and Leave-One-Out baseline
- **Memory management**: three-stage offload-onload of training weights / optimizer / inference weights, single-node RL on large models
- **Multi-turn Agent**: built-in rollout orchestrator, tool registry, sandbox pool, and process reward — agentic RL ready
- **Multi-task rewards**: math, math_dapo, gsm8k, geo3k, code (prime_code), IF (instruction following), etc.

## Architecture Overview

```
                         ┌───────────────────────────────────────────────────┐
                         │                    MyRL Trainer                   │
                         │                                                   │
   ┌──────────────┐      │   ┌────────────┐   ┌────────────┐   ┌──────────┐  │
   │   Dataset    │─────▶│   │ Megatron   │   │  vLLM      │   │ Reference│  │
   │  (jsonl)     │      │   │ Actor 3D   │   │ Engine     │   │  Policy  │  │
   │ prompt+label │      │   │ TP/PP/CP   │◀─▶│ TP-rollout │   │ (frozen) │  │
   └──────────────┘      │   └─────┬──────┘   └─────┬──────┘   └─────┬────┘  │
                         │         │                │                │       │
                         │         │  weights sync  │   rollout      │       │
                         │         └────────┬───────┴────────────────┘       │
                         │                  │                                │
                         │   ┌──────────────▼───────────────┐                │
                         │   │     Memory Manager           │                │
                         │   │  (offload / onload weights & │                │
                         │   │   optimizer states)          │                │
                         │   └──────────────────────────────┘                │
                         │                                                   │
                         │   ┌──────────────┐   ┌────────────────────────┐   │
                         │   │ Reward Score │   │  GRPO Loss             │   │
                         │   │ math/code/IF │   │  clip + KL + TIS       │   │
                         │   └──────────────┘   └────────────────────────┘   │
                         └───────────────────────────┬───────────────────────┘
                                                     │
                              ┌──────────────────────▼─────────────────────┐
                              │         Multi-turn Agent (optional)        │
                              │  Tool Registry │ Sandbox Pool │ Bio Env    │
                              │  Tool Parser   │ Trajectory   │ Reward     │
                              └────────────────────────────────────────────┘
```

Key modules:

- [examples/qwen3/grpo_trainer.py](examples/qwen3/grpo_trainer.py) — GRPO trainer entry point
- [megatron_patch/memory/](megatron_patch/memory/) — trainer / inference memory managers
- [megatron_patch/agent/](megatron_patch/agent/) — multi-turn rollout, tool system, sandbox pool
- [megatron_patch/reward_score/](megatron_patch/reward_score/) — reward functions
- [megatron_patch/rl_utils.py](megatron_patch/rl_utils.py) — KL, advantage, mask and other RL utilities

## Training Workflow

```
  ┌──────────────────────┐
  │ Init: Actor / Ref /  │
  │  vLLM / Dataloader   │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐    each train_iter:
  │ 1. wake up vLLM      │    - offload trainer weights/optimizer
  │    offload trainer   │    - onload inference weights
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐    convert Megatron distributed
  │ 2. weight conversion │    weights to HF layout, load
  │    Megatron → vLLM   │    into vLLM engine on the fly
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐    single-turn:  direct generate
  │ 3. rollout (vLLM)    │    multi-turn:  generate→parse
  │    + reward compute  │                 →tool→append
  └──────────┬───────────┘    reward:  rule-based
             ▼
  ┌──────────────────────┐
  │ 4. sleep vLLM        │
  │    onload trainer    │
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐    actor logprobs (current)
  │ 5. compute logprobs  │    ref   logprobs (frozen)
  │    actor + reference │    via forward-only fwd-bwd
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐    leave-one-out baseline
  │ 6. advantages        │    advantage = (r - μ) / σ
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐    L = clip_loss + β·KL
  │ 7. GRPO update       │    + token-level IS correction
  │    optimizer.step    │    two-level norm (token / group)
  └──────────┬───────────┘
             ▼
        next iteration
```

## Design Philosophy

### 1. Training / Inference Memory Reuse

Holding Actor, Reference, and vLLM weights on the GPU at the same time is impractical for large models. MyRL implements three-stage memory management via [memory_utils.py](megatron_patch/memory_utils.py):

- **Training phase**: Actor weights + optimizer states stay on GPU; Ref and vLLM weights are offloaded to CPU
- **Rollout phase**: vLLM weights onload, trainer weights / optimizer offload, vLLM `wake_up`
- **Logprob phase**: Actor / Ref onload separately, run forward-only, then offload again

`onload / offload` is bucketed by `bucket_size_mb` and combined with `torch.cuda.empty_cache()` and `gc.collect()`, enabling single-node RL on 8B–32B models.

### 2. On-the-Fly Weight Conversion (Megatron → vLLM)

The training side uses Megatron distributed weights (sharded by TP / PP); the inference side uses the HF layout expected by vLLM. `Trainer.convert()` uses `McoreToHFWeightConverterDense` from [convert.py](megatron_patch/convert.py) to produce a per-tensor weight stream and feed it directly into vLLM's `model.load_weights`, avoiding any disk-based conversion.

### 3. GRPO Loss Design

See [grpo_trainer.py:get_actor_forward_output_and_loss_func](examples/qwen3/grpo_trainer.py#L527):

- **Leave-One-Out baseline**: with G rollouts per prompt, each sample uses the mean of the other G-1 as its baseline, normalized by group std
- **PPO-style clipping**: asymmetric clip range `(1-0.2, 1+0.28)` to encourage positive updates
- **Token-level Importance Sampling**: `exp(prev_logprob - generation_logprob)` corrects logprob drift between vLLM and the training backend, clipped at 2.0 (see fengyao's off-policy RL notes)
- **KL penalty**: Joschu 2020 KL estimator added to the actor loss
- **Two-level normalization**: first by per-response length, then by valid samples per group, to keep long / short samples balanced

### 4. Multi-turn Agent Rollout

The `MultiTurnRolloutOrchestrator` in [multi_turn_rollout.py](megatron_patch/agent/multi_turn_rollout.py) implements a generate → parse → execute → append loop:

- **Tool Registry**: pluggable tool registry, ships with python_execute, bio workflow, etc.
- **Sandbox Pool**: process-level sandbox pool with memory / time / CPU limits
- **Trajectory**: records tokens, tool_call, tool_result, and loss_mask per turn — only model-generated tokens contribute to loss
- **Process Reward**: rewards for intermediate steps (e.g. tool-call success) are weighted with the final reward and folded into the advantage

## Quick Start

### 1. Docker Image

```bash
docker pull dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.04
```

### 2. Install vLLM

PyTorch is already in the image — skip its build:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install -r requirements-build.txt
pip install -e . --no-build-isolation
```

### 3. Convert Weights (HF → Mcore)

```bash
cd toolkits/distributed_checkpoints_convertor/
sh scripts/qwen3/run.sh
```

### 4. Prepare Data

JSONL with two fields per line, `prompt` and `label`:

```json
{"prompt": "There were 27 boys and 35 girls on the playground at recess. There were _____ children on the playground at recess.", "label": "62"}
{"prompt": "Find the value of adding 3 to the number of diagonals in the rectangle.", "label": "5"}
```

### 5. Launch Training

```bash
cd examples/qwen3
sh run.sh
```

[examples/qwen3/run.sh](examples/qwen3/run.sh) is the single-node 4-GPU Qwen3-4B sample. Minimal edits:

| Variable | Meaning |
| --- | --- |
| `DATASET_PATH` | training jsonl path |
| `VALID_DATASET_PATH` | validation jsonl path |
| `PRETRAIN_CHECKPOINT_PATH` | Mcore-format weights path |
| `OUTPUT_BASEPATH` | log / checkpoint output root |

## Parameters

### Training Parameters

| Index | Name | Meaning |
| --- | --- | --- |
| $1 | `ENV` | `dsw` single-node / `dlc` multi-node / `tione` |
| $2 | `MODEL_SIZE` | 0.6B / 1.7B / 4B / 8B / 14B / 32B / A3B / A22B |
| $3 | `BATCH_SIZE` | per-DP micro batch |
| $4 | `GLOBAL_BATCH_SIZE` | global batch |
| $5 / $6 | `LR` / `MIN_LR` | learning rate / min learning rate |
| $7 / $8 | `SEQ_LEN` / `PAD_LEN` | sequence length / padding length |
| $9 | `PR` | fp16 / bf16 / fp8 |
| $10–$14 | `TP/PP/CP/ETP/EP` | 3D parallelism + expert parallelism |
| $15 | `SP` | enable sequence parallelism |
| $16 | `DO` | enable ZeRO-1 distributed optimizer |
| $17 | `FL` | prefer Flash Attention |
| $18 | `SFT` | run in SFT mode |
| $19 | `AC` | sel / full / offload / none |
| $20 | `OPTIMIZER_OFFLOAD` | false or 0~1 ratio |
| $21 | `SAVE_INTERVAL` | checkpoint interval |
| $22–$24 | data / valid / pretrain paths | |
| $25 / $26 | `TRAIN_ITERS` / `WARMUP_ITERS` | train / warmup iters (or tokens) |
| $27 | `OUTPUT_BASEPATH` | output directory |

### RL Parameters

```bash
--gpu-memory-utilization 0.6        # vLLM GPU memory fraction
--vllm-max-model-len      16384     # vLLM max tokens
--vllm-tensor-parallel-size 2       # vLLM TP
--vllm-max-num-batched-tokens 8192
--vllm-temperature 1.0              # rollout temperature
--vllm-top-p 1.0
--vllm-max-new-tokens 8192          # max tokens per rollout
--vllm-num-rollout-samples 8        # samples per prompt (G)
--kl-penalty 0.001                  # β
```

### Agent Parameters (optional)

```bash
--agent-multi-turn                  # enable multi-turn agent rollout
--agent-tool-format hermes          # tool-call format
--agent-max-turns 5                 # max turns per trajectory
--agent-max-total-tokens 16384
--sandbox-pool-size 16
--sandbox-max-memory-mb 1024
--sandbox-timeout 30
--final-reward-weight 1.0
--process-reward-weight 0.1
--tool-success-reward 0.05
--tool-failure-penalty -0.05
```

## Benchmarks

> Numbers below are from internal testing (H800 / A100). Actual results vary with hardware, network, and dataset. Cells marked `—` are still being filled in.

### Model Support Matrix

| Model | Params | Architecture | Recommended TP / PP / CP | Recommended ETP / EP |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 0.6B | Dense | 1 / 1 / 1 | — |
| Qwen3-1.7B | 1.7B | Dense | 1 / 1 / 1 | — |
| Qwen3-4B | 4B | Dense | 4 / 1 / 1 | — |
| Qwen3-8B | 8B | Dense | 4 / 1 / 1 | — |
| Qwen3-14B | 14B | Dense | 4 / 2 / 1 | — |
| Qwen3-32B | 32B | Dense | 8 / 2 / 1 | — |
| Qwen3-A3B | 30B(A3B) | MoE | 4 / 1 / 1 | 1 / 4 |
| Qwen3-A22B | 235B(A22B) | MoE | 8 / 4 / 1 | 1 / 8 |

### Training Throughput (TBD)

Conditions: `SEQ_LEN=8192`, `PAD_LEN=8192`, `bf16`, `vllm_num_rollout_samples=8`, GRPO. One step = rollout + train.

| Model | GPU | TP/PP | rollout (s/step) | train (s/step) | total (s/step) | tokens/s/GPU |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B | 4×H800 | 4 / 1 | — | — | — | — |
| Qwen3-8B | 8×H800 | 4 / 1 | — | — | — | — |
| Qwen3-14B | 8×H800 | 4 / 2 | — | — | — | — |
| Qwen3-32B | 16×H800 | 8 / 2 | — | — | — | — |
| Qwen3-A3B | 8×H800 | 4 / 1 | — | — | — | — |

### RL Effectiveness (GSM8K / MATH, TBD)

| Model | Initial reward | 1k step | 3k step | Converged reward |
| --- | --- | --- | --- | --- |
| Qwen3-4B | — | — | — | — |
| Qwen3-8B | — | — | — | — |
| Qwen3-14B | — | — | — | — |

### Agent Task Effectiveness (TBD)

| Task | Model | Initial success | Converged success | Avg turns | Tool success |
| --- | --- | --- | --- | --- | --- |
| Bio Workflow | Qwen3-8B | — | — | — | — |
| Math + Code | Qwen3-8B | — | — | — | — |

## Roadmap

- [ ] More dense models: Llama, Mistral, Gemma
- [ ] Full MoE training (end-to-end validation for A3B / A22B)
- [ ] More algorithms: PPO, GSPO, DAPO, ReMax
- [ ] Multi-task reward weighting
- [ ] Official benchmarks with reproducible scripts

## References

- [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)
- [Verl](https://github.com/volcengine/verl)
- [ChatLearn](https://github.com/alibaba/ChatLearn)
- [vLLM](https://github.com/vllm-project/vllm)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
