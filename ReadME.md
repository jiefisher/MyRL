<div align="center">
 👋 MyRL -- A Lightweight Reinforcement Learning Library
    <br>
    <br>
</div>

MyRL is an open-source, high-performance, lightweight RLHF library built on **vLLM** and **Megatron-LM**. It supports multi-node multi-GPU training, 3D parallelism, and utilizes vLLM to accelerate inference generation.

## New Features
- Megatron-LM
- VLLM
- 3D Parallelism

## Getting Started

### Docker Image
```bash
dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.04
```

### VLLM Installation

PyTorch is already included in the image, so you need to skip its installation.
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install -r requirements-build.txt
pip install -e . --no-build-isolation
```

### Usage

#### Conversion

Before running, you need to convert the model from HuggingFace format to Mcore format.

```bash
cd toolkits/distributed_checkpoints_convertor/
sh scripts/qwen3/run.sh
```

#### Prepare Data

The data format consists of JSONL strings containing two key fields: **prompt** and **label**.

Example:
```json
{"prompt": "There were 27 boys and 35 girls on the playground at recess. There were _____ children on the playground at recess.", "label": "62"}
{"prompt": "Find the value of adding 3 to the number of diagonals in the rectangle.", "label": "5"}
```

#### Model Training

```bash
cd examples/qwen3
sh run.sh
```

#### Unified Description of Fine-tuning Commands

The list of required parameters is as follows:
```bash
ENV=$1                          # Runtime environment switch: 'dsw' for single-node training, 'dlc' for multi-node training
MODEL_SIZE=$2                   # Model size scale: 0.6B, 1.7B, 4B, 8B, 14B, 32B, A3B, A22B
BATCH_SIZE=$3                   # Number of samples per data parallel rank in one iteration
GLOBAL_BATCH_SIZE=$4            # Total number of samples across all data parallel ranks in one iteration
LR=$5                           # Learning rate
MIN_LR=$6                       # Minimum learning rate
SEQ_LEN=$7                      # Sequence length
PAD_LEN=$8                      # Padding length
PR=${9}                         # Training precision: fp16, bf16, fp8
TP=${10}                        # Tensor parallelism degree
PP=${11}                        # Pipeline parallelism degree
CP=${12}                        # Context parallelism degree
ETP=${13}                       # Expert tensor parallelism degree
EP=${14}                        # Expert parallelism degree
SP=${15}                        # Whether to use sequence parallelism: true, false
DO=${16}                        # Whether to use Megatron version of Zero-1 memory optimizer: true, false
FL=${17}                        # Whether to prioritize Flash Attention: true, false
SFT=${18}                       # Whether to perform fine-tuning (SFT): true, false
AC=${19}                        # Activation checkpointing mode: sel, full, offload, false
OPTIMIZER_OFFLOAD=${20}         # Whether to enable Optimizer Offload: false, or input a decimal between 0-1 as the offload ratio
SAVE_INTERVAL=${21}             # Checkpoint saving interval
DATASET_PATH=${22}              # Training dataset path
VALID_DATASET_PATH=${23}        # Validation dataset path
PRETRAIN_CHECKPOINT_PATH=${24}  # Pre-trained model path
TRAIN_TOKENS_OR_ITERS=${25}     # Number of training Tokens or Iters
WARMUP_TOKENS_OR_ITERS=${26}    # Number of warmup Tokens or Iters
OUTPUT_BASEPATH=${27}           # Training output log file path
```

RL parameters are as follows:
```bash
--gpu-memory-utilization 0.6 \             # vLLM GPU utilization
--vllm-max-model-len 16384 \               # vLLM max token count
--vllm-tensor-parallel-size 2 \            # vLLM model parallelism degree
--vllm-max-num-batched-tokens 8192 \       
--vllm-temperature 1.0 \                   # Temperature during rollout
--vllm-top-p 1.0 \                         # Top-p during rollout
--vllm-max-new-tokens 8192 \               # Max new tokens generated during rollout
--vllm-num-rollout-samples 8 \             # Number of rollout samples
--kl-penalty 0.001                         # KL penalty
```

## Coming Soon
- Support for more Dense models, such as Llama, Mistral, Gemma, etc.
- Support for MoE architecture.
- Support for multiple algorithms, such as PPO, GSPO, DAPO, etc.

## References
- [Pai-Megatron_patch](https://github.com/alibaba/Pai-Megatron-Patch)
- [Verl](https://github.com/volcengine/verl)
- [ChatLearn](https://github.com/alibaba/ChatLearn)