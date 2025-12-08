<div align="center">
 👋 MyRL -- 一个轻量的强化学习代码库
    <br>
    <br>
</div>
MyRL 是一个基于 vLLM、Megatron-LM 构建的开源高性能轻量级 RLHF 框架，支持多级多卡，3D 并行，并使用 vLLM 加速推理生成

## 新特性
- Megatron-LM
- VLLM
- 3D并行
## 开始
### 安装镜像
```bash
dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.04
```

### VLLM安装

镜像中已包含pytorch，需要跳过安装
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install -r requirements-build.txt
pip install -e . --no-build-isolation
```

### 运行
#### 转换

运行时需要先将模型从Huggingface格式到Mcore格式的转换

cd toolkits/distributed_checkpoints_convertor/
sh scripts/qwen3/run.sh

####准备数据

数据格式为jsonl的字符串，包含两个关键的Key：**prompt** 及 **label**

例如：
```json
{"prompt": "There were 27 boys and 35 girls on the playground at recess. There were _____ children on the playground at recess.", "label": "62"}
{"prompt": "Find the value of adding 3 to the number of diagonals in the rectangle.", "label": "5"}
```
#### 模型训练

```bash
cd examples/qwen3
sh run.sh
```
#### 微调命令统一描述

需要传入的参数列表如下：
```bash
ENV=$1                          # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
MODEL_SIZE=$2                   # 模型结构参数量级: 0.6B, 1.7B, 4B, 8B, 14B, 32B, A3B, A22B
BATCH_SIZE=$3                   # 一次迭代一个数据并行内的样本数
GLOBAL_BATCH_SIZE=$4            # 一次迭代多个数据并行的总样本数
LR=$5                           # 学习率
MIN_LR=$6                       # 最小学习率
SEQ_LEN=$7                      # 序列长度
PAD_LEN=$8                      # Padding长度
PR=${9}                         # 训练精度: fp16, bf16, fp8
TP=${10}                        # 模型并行度
PP=${11}                        # 流水并行度
CP=${12}                        # 上下文并行度
ETP=${13}                       # 专家张量并行度
EP=${14}                        # 专家模型并行度
SP=${15}                        # 是否使用序列并行: true, false
DO=${16}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${17}                        # 是否优先使用Flash Attention: true, false
SFT=${18}                       # 是否执行微调训练: true, false
AC=${19}                        # 激活检查点模式: sel, full, offload, false
OPTIMIZER_OFFLOAD=${20}         # 是否启用Offload optimizer: false, 或输入0～1的小数作为参数offload比例
SAVE_INTERVAL=${21}             # 保存ckpt的间隔
DATASET_PATH=${22}              # 训练数据集路径
VALID_DATASET_PATH=${23}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${24}  # 预训练模型路径
TRAIN_TOKENS_OR_ITERS=${25}     # 训练TOKEN或者Iter数
WARMUP_TOKENS_OR_ITERS=${26}    # 预热TOKEN或者Iter数        
OUTPUT_BASEPATH=${27}           # 训练输出日志文件路径
```
RL 参数如下
```bash
--gpu-memory-utilization 0.6 \             # vllmGPU使用量
--vllm-max-model-len 16384 \               # vllm最大token数量
--vllm-tensor-parallel-size 2 \            # vllm模型并行度
--vllm-max-num-batched-tokens 8192 \       
--vllm-temperature 1.0 \                   # rollout时的温度
--vllm-top-p 1.0 \                         # rollout时的top-p
--vllm-max-new-tokens 8192 \               # rollout时的最大token生成数量
--vllm-num-rollout-samples 8 \             # rollout数量
--kl-penalty 0.001                         # kl惩罚
```
## 即将推出
- 支持更多Dense模型，如Llama，Mistral，Gemma等
- 支持MoE架构
- 支持多种算法，如PPO，GSPO， DAPO等
## 参考
- [Pai-Megatron_patch](https://github.com/alibaba/Pai-Megatron-Patch)
- [Verl](https://github.com/volcengine/verl)
- [ChatLearn](https://github.com/alibaba/ChatLearn)