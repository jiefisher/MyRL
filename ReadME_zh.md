<div align="center">
👋 MyRL — 一个轻量级强化学习训练框架
<br>
<br>
</div>

MyRL 是一个基于 **Megatron-LM** + **vLLM** 构建的开源高性能轻量级 RLHF / RL 训练框架。聚焦于大模型场景下的强化学习训练，原生支持 3D 并行（TP / PP / CP）、MoE 专家并行（ETP / EP），并通过 vLLM 加速 rollout 阶段的推理生成。同时内置 multi-turn agent rollout、工具调用沙箱、过程奖励等能力，便于快速搭建 agentic RL 训练流水线。

## 目录

- [核心特性](#核心特性)
- [架构总览](#架构总览)
- [训练流程](#训练流程)
- [设计思路](#设计思路)
- [快速开始](#快速开始)
- [参数说明](#参数说明)
- [跑分与性能](#跑分与性能)
- [Roadmap](#roadmap)
- [参考](#参考)

## 核心特性

- **训练后端**：基于 Megatron-LM，原生支持 TP / PP / CP / ETP / EP，覆盖 Dense 与 MoE 模型
- **推理后端**：基于 vLLM，支持 sleep / wake-up 模式，rollout 阶段独占显存，训练阶段释放
- **算法**：GRPO（带 KL 惩罚、PPO-style clipping、Token-level Importance Sampling、Leave-One-Out baseline）
- **显存管理**：训练权重 / 优化器 / 推理权重三段式 offload-onload，单机即可完成大模型 RL
- **Multi-turn Agent**：内置 rollout orchestrator、工具注册表、沙箱池与过程奖励，支持 agentic RL
- **多任务奖励**：内置 math、math_dapo、gsm8k、geo3k、code (prime_code)、IF (instruction following) 等 reward 函数

## 架构总览

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
                              │         Multi-turn Agent (可选)            │
                              │  Tool Registry │ Sandbox Pool │ Bio Env    │
                              │  Tool Parser   │ Trajectory   │ Reward     │
                              └────────────────────────────────────────────┘
```

关键模块：

- [examples/qwen3/grpo_trainer.py](examples/qwen3/grpo_trainer.py) — GRPO 训练器主入口
- [megatron_patch/memory/](megatron_patch/memory/) — Trainer / Inference 显存管理
- [megatron_patch/agent/](megatron_patch/agent/) — 多轮 rollout、工具系统、沙箱池
- [megatron_patch/reward_score/](megatron_patch/reward_score/) — 奖励函数集
- [megatron_patch/rl_utils.py](megatron_patch/rl_utils.py) — KL、advantage、mask 等 RL 工具

## 训练流程

```
  ┌──────────────────────┐
  │ Init: Actor / Ref /  │
  │  vLLM / Dataloader   │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐    每个 train_iter:
  │ 1. wake up vLLM      │    - 训练权重/优化器 offload
  │    offload trainer   │    - 推理权重 onload
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐    将 Megatron 分布式权重
  │ 2. weight conversion │    转换为 HF layout，在线
  │    Megatron → vLLM   │    加载到 vLLM 引擎
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐    single-turn:  直接 generate
  │ 3. rollout (vLLM)    │    multi-turn:  generate→parse
  │    + reward compute  │                  →tool→append
  └──────────┬───────────┘    reward:  rule-based
             ▼
  ┌──────────────────────┐
  │ 4. sleep vLLM        │
  │    onload trainer    │
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐    actor logprobs (current)
  │ 5. compute logprobs  │    ref   logprobs (frozen)
  │    actor + reference │    用 forward-only fwd-bwd
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐    leave-one-out baseline
  │ 6. advantages        │    advantage = (r - μ) / σ
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐    L = clip_loss + β·KL
  │ 7. GRPO update       │    + token-level IS 校正
  │    optimizer.step    │    两级归一化 (token / group)
  └──────────┬───────────┘
             ▼
        next iteration
```

## 设计思路

### 1. 训练 / 推理显存复用

GPU 同时承载 Actor、Reference、vLLM 三份模型权重在大模型场景下不现实。MyRL 通过 [memory_utils.py](megatron_patch/memory_utils.py) 实现三段式显存管理：

- **训练阶段**：Actor 权重 + 优化器状态在 GPU，Ref 权重和 vLLM 权重 offload 到 CPU
- **Rollout 阶段**：vLLM 权重 onload，训练权重 / 优化器 offload，vLLM `wake_up`
- **logprobs 计算**：Actor / Ref 分别 onload，做 forward-only 计算后再 offload

`onload / offload` 以 bucket_size_mb 为单位调度，配合 `torch.cuda.empty_cache()` 与 `gc.collect()`，在 8B-32B 模型上可单机完成 RL 训练。

### 2. 权重在线转换（Megatron → vLLM）

训练侧使用 Megatron 的分布式权重（按 TP / PP 切分），推理侧使用 vLLM 期望的 HF layout。`Trainer.convert()` 通过 [convert.py](megatron_patch/convert.py) 中的 `McoreToHFWeightConverterDense` 在线生成 per-tensor 权重流，直接喂给 vLLM 的 `model.load_weights`，避免落盘转换。

### 3. GRPO Loss 设计

参见 [grpo_trainer.py:get_actor_forward_output_and_loss_func](examples/qwen3/grpo_trainer.py#L527)：

- **Leave-One-Out baseline**：同一个 prompt 下 G 个 rollout，对每个样本用其余 G-1 个的均值作为 baseline，方差用组内 std 归一化
- **PPO-style clipping**：非对称 clip `(1-0.2, 1+0.28)`，鼓励正向更新
- **Token-level Importance Sampling**：`exp(prev_logprob - generation_logprob)` 修正 vLLM 与训练后端 logprob 偏差，clip 上界 2.0（参考 fengyao 的 off-policy RL 笔记）
- **KL 惩罚**：Joschu 2020 形式 KL 估计，加在 actor loss 上
- **两级归一化**：先按响应长度归一化，再按组内有效样本数归一化，避免长 / 短样本梯度失衡

### 4. Multi-turn Agent Rollout

[multi_turn_rollout.py](megatron_patch/agent/multi_turn_rollout.py) 中的 `MultiTurnRolloutOrchestrator` 实现 generate → parse → execute → append 循环：

- **Tool Registry**：可插拔工具注册表，内置 python_execute、bio workflow 等
- **Sandbox Pool**：进程级沙箱池，限制内存 / 时间 / CPU
- **Trajectory**：记录每一轮的 token、tool_call、tool_result 与 loss_mask，仅对模型生成的 token 计 loss
- **Process Reward**：对中间步骤（如工具调用成功率）给出过程奖励，与 final reward 加权合并到 advantage

## 快速开始

### 1. 镜像

```bash
docker pull dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.04
```

### 2. 安装 vLLM

镜像中已含 PyTorch，编译时跳过：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install -r requirements-build.txt
pip install -e . --no-build-isolation
```

### 3. 转换权重 (HF → Mcore)

```bash
cd toolkits/distributed_checkpoints_convertor/
sh scripts/qwen3/run.sh
```

### 4. 准备数据

数据为 jsonl，每行包含 `prompt` 与 `label` 两个字段：

```json
{"prompt": "There were 27 boys and 35 girls on the playground at recess. There were _____ children on the playground at recess.", "label": "62"}
{"prompt": "Find the value of adding 3 to the number of diagonals in the rectangle.", "label": "5"}
```

### 5. 启动训练

```bash
cd examples/qwen3
sh run.sh
```

[examples/qwen3/run.sh](examples/qwen3/run.sh) 是单机 4 卡 Qwen3-4B 的样例，最小修改：

| 变量 | 含义 |
| --- | --- |
| `DATASET_PATH` | 训练集 jsonl 路径 |
| `VALID_DATASET_PATH` | 验证集 jsonl 路径 |
| `PRETRAIN_CHECKPOINT_PATH` | Mcore 格式权重路径 |
| `OUTPUT_BASEPATH` | 日志 / checkpoint 输出根目录 |

## 参数说明

### 训练参数

| 序号 | 名称 | 含义 |
| --- | --- | --- |
| $1 | `ENV` | `dsw` 单机 / `dlc` 多机 / `tione` |
| $2 | `MODEL_SIZE` | 0.6B / 1.7B / 4B / 8B / 14B / 32B / A3B / A22B |
| $3 | `BATCH_SIZE` | DP 内 micro batch |
| $4 | `GLOBAL_BATCH_SIZE` | 全局 batch |
| $5 / $6 | `LR` / `MIN_LR` | 学习率 / 最小学习率 |
| $7 / $8 | `SEQ_LEN` / `PAD_LEN` | 序列长度 / padding 长度 |
| $9 | `PR` | fp16 / bf16 / fp8 |
| $10–$14 | `TP/PP/CP/ETP/EP` | 3D 并行 + 专家并行 |
| $15 | `SP` | 是否开启序列并行 |
| $16 | `DO` | 是否使用 ZeRO-1 优化器 |
| $17 | `FL` | 是否优先 Flash Attention |
| $18 | `SFT` | 是否 SFT 模式 |
| $19 | `AC` | sel / full / offload / none |
| $20 | `OPTIMIZER_OFFLOAD` | false 或 0~1 比例 |
| $21 | `SAVE_INTERVAL` | checkpoint 间隔 |
| $22–$24 | 数据 / 验证 / 预训练 路径 | |
| $25 / $26 | `TRAIN_ITERS` / `WARMUP_ITERS` | 训练 / 预热步数（或 token 数） |
| $27 | `OUTPUT_BASEPATH` | 输出目录 |

### RL 参数

```bash
--gpu-memory-utilization 0.6        # vLLM 显存占用比例
--vllm-max-model-len      16384     # vLLM 最大 token
--vllm-tensor-parallel-size 2       # vLLM TP
--vllm-max-num-batched-tokens 8192
--vllm-temperature 1.0              # rollout 温度
--vllm-top-p 1.0
--vllm-max-new-tokens 8192          # 单次最长生成
--vllm-num-rollout-samples 8        # 每个 prompt 采样数 (G)
--kl-penalty 0.001                  # β
```

### Agent 参数（可选）

```bash
--agent-multi-turn                  # 启用多轮 agent rollout
--agent-tool-format hermes          # tool call 格式
--agent-max-turns 5                 # 单次 trajectory 最大轮数
--agent-max-total-tokens 16384
--sandbox-pool-size 16
--sandbox-max-memory-mb 1024
--sandbox-timeout 30
--final-reward-weight 1.0
--process-reward-weight 0.1
--tool-success-reward 0.05
--tool-failure-penalty -0.05
```


## Roadmap

- [ ] 支持更多 Dense 模型：Llama、Mistral、Gemma
- [ ] 完善 MoE 训练（A3B / A22B 全流程验证）
- [ ] 支持更多算法：PPO、GSPO、DAPO、ReMax
- [ ] 多任务 reward 加权调度
- [ ] 提供官方跑分与可复现脚本

## 参考

- [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)
- [Verl](https://github.com/volcengine/verl)
- [ChatLearn](https://github.com/alibaba/ChatLearn)
- [vLLM](https://github.com/vllm-project/vllm)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
