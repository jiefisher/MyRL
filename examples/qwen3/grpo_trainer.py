from typing import Union
from collections import UserDict
from contextlib import nullcontext
import torch
import torch._dynamo
import inspect
import torch.distributed as dist
from typing_extensions import Self
import gc
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron_patch.tokenizer import build_tokenizer
from megatron_patch.model.qwen3_moe.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core import mpu
from megatron.core.transformer.spec_utils import import_module
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron_patch.data import train_valid_test_datasets_provider
from megatron_patch.rl_utils import (
    create_mask,
    masked_mean, 
    calculate_kl_penalty_joschu2020,
    calculate_baseline_and_std_per_prompt,
)

from megatron.training import get_args, print_rank_0
from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.utils import (
    calc_params_l2_norm,

)
torch._dynamo.config.suppress_errors = True
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import set_jit_fusion_options
from megatron.training import get_model, ft_integration
from megatron.training.checkpointing import load_checkpoint

from vllm import LLM, SamplingParams
from megatron_patch.training import setup_model_and_optimizer
from megatron_patch.utils import per_tensor_generator
from megatron_patch.convert import McoreToHFWeightConverterDense
from transformers import AutoConfig
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
)

from megatron.legacy.data.data_samplers import build_dataloader
from megatron.core.utils import divide
from transformers import AutoTokenizer

from typing import Dict, List, Optional, Union, Callable
from vllm import LLM, SamplingParams
from megatron_patch.distributed import broadcast_2d_tensor_within_mp, broadcast_tensor_within_pp, rebalance_nd_tensor, from_parallel_logits_to_logprobs, broadcast_2d_tensor_within_pp
from megatron_patch.reward_score.rpf import compute_score
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron_patch.distributed import get_iterator_k_split
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.global_vars import get_timers
from megatron.core.num_microbatches_calculator import (
    get_num_microbatches,
    update_num_microbatches)
from megatron_patch.data_utils import *
from megatron_patch.convert import McoreToHFWeightConverterDense
from megatron_patch.memory_utils import create_trainer_memory_manager, create_inference_memory_manager

from megatron.training.checkpointing import save_checkpoint
# from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
# from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


        
def get_last_rank():
    return torch.distributed.get_world_size() - 1

def get_forward_output_only_func():
    def fwd_output_only_func(dataloader_iter, model):
        # If tuple, 1st element in it is the batch since dataloader_iter returns batch, batch_idx, dataloader_idx
        batch = next(dataloader_iter)
        if isinstance(batch, tuple):
            batch = batch[0]
        extra_arg = {}
        if len(batch) == 3:
            batch = [x.cuda(non_blocking=True) for x in batch]
            tokens, attention_mask, position_ids = batch
        
        output_tensor = model(tokens, position_ids, attention_mask, **extra_arg)

        def id_func(output_tensor):
            return output_tensor, {'logits': output_tensor}

        return output_tensor, id_func

    return fwd_output_only_func
        
def get_logprob_output_only_func(inference_only=True):
    fwd_output_only_func = get_forward_output_only_func()

    def log_prob_output_only_func(dataloader_iter, model):
        batch = next(dataloader_iter)

        output_tensor, _ = fwd_output_only_func(iter([batch,]), model)

        def id_func(output_tensor, non_loss_data=True):
            logprobs = from_parallel_logits_to_logprobs(
                vocab_parallel_logits=output_tensor,
                target=batch[0],
                inference_only=inference_only,
                higher_stability=True,
            )
            return logprobs

        return output_tensor, id_func

    return log_prob_output_only_func

def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x
            
class Trainer:
    def __init__(self):
        from megatron_patch.arguments import get_patch_args
        initialize_megatron(extra_args_provider=get_patch_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
        args = get_args()
        args.iteration = 0
        self.model, self.optimizer, self.opt_param_scheduler = setup_model_and_optimizer(
        self.model_provider, ModelType.encoder_or_decoder)
        self._memory_manager = create_trainer_memory_manager(
            self.model,
            self.optimizer,
            1024,
        )
        
        self._memory_manager.offloads()
        
        self.ref_model = get_model(self.model_provider,
                          model_type=ModelType.encoder_or_decoder,
                          wrap_with_ddp=False)
        
        if args.load is not None:
            ref_load_path = getattr(args, 'ref_load', None) or args.load
            original_load = args.load
            args.load = ref_load_path
            load_checkpoint(self.ref_model, None, None)
            args.load = original_load
            
        for model_chunk in self.ref_model:
            model_chunk.eval()

        self.ref_memory_manager = create_inference_memory_manager(
            self.ref_model,
            1024,
            )
        
        self.ref_memory_manager.offloads()

        self.tokenizer = AutoTokenizer.from_pretrained(
                        args.load,
                        padding_side="right",
                        use_fast=False,
                        trust_remote_code=True
                    )
        
        self.inference_engine = LLM(
            model= args.load,
            enable_sleep_mode=True,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype="bfloat16",
            enforce_eager=False,
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=args.vllm_max_model_len,
            load_format="dummy",
            disable_log_stats=True,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            trust_remote_code=True,
            seed=args.seed,
        )
        self.sampling_params = SamplingParams(temperature=args.vllm_temperature, top_p=args.vllm_top_p,max_tokens=args.vllm_max_new_tokens,logprobs=1)
        self.inference_engine.sleep(2)
        print_rank_0("inference loaded")
        collate_fn = torch.utils.data.dataloader.default_collate
        print_rank_0("data loading")
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = \
        self.build_train_valid_test_data_loaders(
            train_valid_test_datasets_provider,collate_fn)
        print_rank_0("data get")
        self.step = 0
        print_rank_0("init finfish")

        # Agent multi-turn initialization
        if getattr(args, 'agent_multi_turn', False):
            from megatron_patch.agent import (
                ToolRegistry, ToolCallParser, ToolFormatter,
                SandboxPool, SandboxConfig, MultiTurnRolloutOrchestrator,
                AgentRewardComputer, BioToolExecutor, register_bio_tools,
            )
            self.tool_registry = ToolRegistry()
            self.tool_registry.register_builtin_tools()
            register_bio_tools(self.tool_registry)
            self.tool_parser = ToolCallParser(format=args.agent_tool_format)
            self.tool_formatter = ToolFormatter(self.tokenizer, format=args.agent_tool_format)

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                self.sandbox_pool = SandboxPool(SandboxConfig(
                    pool_size=args.sandbox_pool_size,
                    max_memory_mb=args.sandbox_max_memory_mb,
                    max_wall_time_sec=args.sandbox_timeout,
                ))
                self.bio_tool_executor = BioToolExecutor()
            else:
                self.sandbox_pool = None
                self.bio_tool_executor = None

            self.multi_turn_orchestrator = MultiTurnRolloutOrchestrator(
                inference_engine=self.inference_engine,
                sampling_params=self.sampling_params,
                tokenizer=self.tokenizer,
                tool_registry=self.tool_registry,
                tool_parser=self.tool_parser,
                tool_formatter=self.tool_formatter,
                sandbox_pool=self.sandbox_pool,
                max_turns=args.agent_max_turns,
                max_total_tokens=args.agent_max_total_tokens,
                bio_tool_executor=self.bio_tool_executor,
            )
            self.agent_reward = AgentRewardComputer(
                final_reward_weight=args.final_reward_weight,
                process_reward_weight=args.process_reward_weight,
                tool_success_reward=args.tool_success_reward,
                tool_failure_penalty=args.tool_failure_penalty,
            )
            print_rank_0("agent multi-turn initialized")

    def get_train_valid_test_num_samples(self):
        """Train/valid/test num samples."""
    
        
        # Number of train/valid/test samples.
        args = get_args()
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
    
        
        # eval_iters = (args.train_iters // args.eval_interval + 1) * \
        #              args.eval_iters
        eval_iters = args.eval_iters
        test_iters = args.eval_iters
    
        return (
            train_samples,
            eval_iters * args.global_batch_size,
            test_iters * args.global_batch_size,
        )
    
    def build_train_valid_test_datasets(self, build_train_valid_test_datasets_provider):
        """Build pretraining datasets."""
        train_valid_test_num_samples = self.get_train_valid_test_num_samples()
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_valid_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_valid_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_valid_test_num_samples[2]))
        return build_train_valid_test_datasets_provider(train_valid_test_num_samples)

        
    def build_train_valid_test_data_loaders(
        self, build_train_valid_test_datasets_provider,collate_fn):
        """Build pretraining data loaders."""
    
        (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)
    
        print_rank_0('> building train, validation, and test datasets ...')
        args = get_args()
        # Backward compatibility, assume fixed batch size.
        if args.iteration > 0 and args.consumed_train_samples == 0:
            assert args.train_samples is None, \
                'Only backward compatiblity support for iteration-based training'
            args.consumed_train_samples = args.iteration * args.global_batch_size
        if args.iteration > 0 and args.consumed_valid_samples == 0:
            if args.train_samples is None:
                args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                    args.eval_iters * args.global_batch_size
    
        # Rely on distributed-aware core datasets, temporary
        is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)
    
        # Construct the data pipeline
        if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:
            # Build datasets.
            train_ds, valid_ds, test_ds = self.build_train_valid_test_datasets(
                build_train_valid_test_datasets_provider)
    
            # Build dataloders.
            train_dataloader = build_dataloader(
                dataset=train_ds,
                consumed_samples=args.consumed_train_samples,
                mbs=1, #cfg.model.reinforce.rollout_micro_batch_size
                gbs=args.global_batch_size, #cfg.model.reinforce.num_rollout_samples
                collate_fn=collate_fn,
                load_gbs=False,
                )
            if args.skip_train:
                valid_dataloader = build_dataloader(
                    dataset=valid_ds,
                    consumed_samples=0,
                    mbs=1, #cfg.model.reinforce.rollout_micro_batch_size
                    gbs=args.global_batch_size, #cfg.model.reinforce.num_rollout_samples
                    collate_fn=collate_fn,
                    load_gbs=False,
                    use_random_sampler=False,
                    )
            else:
                valid_dataloader = build_dataloader(
                    dataset=valid_ds,
                    consumed_samples=args.consumed_valid_samples,
                    mbs=1, #cfg.model.reinforce.rollout_micro_batch_size
                    gbs=args.global_batch_size, #cfg.model.reinforce.num_rollout_samples
                    collate_fn=collate_fn,
                    load_gbs=False,
                    use_random_sampler=False,
                    )
            test_dataloader = valid_dataloader
    
            # Flags to know if we need to do training/validation/testing.
            do_train = train_dataloader is not None and args.train_iters > 0
            do_valid = valid_dataloader is not None and args.eval_iters > 0
            do_test = test_dataloader is not None and args.eval_iters > 0
            flags = torch.tensor(
                [int(do_train), int(do_valid), int(do_test)],
                dtype=torch.long, device='cuda')
        else:
            flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')
    
        torch.distributed.broadcast(flags, 0)
    
        args.do_train = getattr(args, "do_train", False) or flags[0].item()
        args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
        args.do_test = getattr(args, "do_test", False) or flags[2].item()
    
        return train_dataloader, valid_dataloader, test_dataloader
        
    def model_provider(self, pre_process=True, post_process=True) -> Union[GPTModel]:
        """Builds the model.
    
        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.
    
        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
    
    
        Returns:
            Union[GPTModel]: The returned model
        """
        args = get_args()
        build_tokenizer(args)
        use_te = args.transformer_impl == "transformer_engine"
    
        if args.record_memory_history:
            torch.cuda.memory._record_memory_history(True,
                # keep 100,000 alloc/free events from before the snapshot
                trace_alloc_max_entries=100000,
    
                # record stack information for the trace events
                trace_alloc_record_context=True)
    
            def oom_observer(device, alloc, device_alloc, device_free):
                # snapshot right after an OOM happened
                print('saving allocated state during OOM')
                snapshot = torch.cuda.memory._snapshot()
                from pickle import dump
                dump(snapshot, open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'))
    
            torch._C._cuda_attach_out_of_memory_observer(oom_observer)
    
        print_rank_0('building QWen3 model ...')
        # Experimental loading arguments from yaml
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)
    
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te, normalization=args.normalization)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm,
                        normalization=args.normalization)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)
    
        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init
    
                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True
    
                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")
    
        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling,
                mtp_block_spec=mtp_block_spec,
            )

        return model

    @torch.no_grad()
    def convert(self):
        # no_grad = torch.no_grad()
        # no_grad.__enter__()
        args = get_args()
        torch.distributed.barrier()
        self._memory_manager.onload_weights()
        transformer_config = core_transformer_config_from_args(args)
        model_config = AutoConfig.from_pretrained(args.load)
        weight_converter = McoreToHFWeightConverterDense(model_config, transformer_config)
        layer_name_mapping = {
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.weight",
        }
        per_tensor_param = per_tensor_generator(
                    self.model,
                    model_config,
                    weight_converter,
                    transformer_config,
                    layer_name_mapping,
        )
        vllm_model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        loaded_params = vllm_model.load_weights(per_tensor_param)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        self._memory_manager.offload_weights()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        del per_tensor_param
        del loaded_params
        torch._C._cuda_clearCublasWorkspaces()
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
        # no_grad.__exit__(None, None, None)
    
    @torch.no_grad()
    def get_inference_log_probs(self, model, response_tokens, eos_id, forward_micro_batch_size=None):
        args = get_args()
        if forward_micro_batch_size is None:
            forward_micro_batch_size = args.micro_batch_size
            
        mbs, seq_length = response_tokens.size()
        num_microbatches = divide(mbs, forward_micro_batch_size)
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(response_tokens, self.tokenizer.eos_token_id, False, False, False)
        attention_mask = attention_mask.expand(response_tokens.size(0), -1, -1, -1)
        position_ids = position_ids.expand(response_tokens.size(0), -1)
    
        batch_iter = get_iterator_k_split([response_tokens, attention_mask, position_ids], num_microbatches)
    
        fwd_bwd_function = get_forward_backward_func()
        logprobs_list = fwd_bwd_function(
            forward_step_func=get_logprob_output_only_func(inference_only=True),
            data_iterator=batch_iter,
            model=model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=forward_micro_batch_size,
            collect_non_loss_data=True,
        )
    
        logprobs = torch.cat(logprobs_list) if len(logprobs_list) > 0 else None
    
        # Broadcast it from last PP stage to everything else.
        logprobs = broadcast_2d_tensor_within_pp(logprobs)
    
        return logprobs

    def get_actor_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(data_iterator, model):
            args = get_args()
            batch = next(data_iterator)
            required_keys = set()
            if mpu.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add("attention_mask")
    
                if mpu.is_pipeline_first_stage():
                    required_keys.update(("response_tokens", "position_ids"))
    
                if mpu.is_pipeline_last_stage():
                    required_keys.update(("response_tokens", "advantages", "mask", "prev_logprobs", "reference_policy_logprobs", "is_end"))
                    # Agent multi-turn keys
                    if getattr(args, 'agent_multi_turn', False):
                        required_keys.update(("agent_loss_mask", "process_rewards"))
    
            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}
    
            parallel_logits = model(
                batch["response_tokens"], batch["position_ids"], batch["attention_mask"], labels=None,
            )
    
            def loss_func(parallel_logits):
                args = get_args()
                mask = batch["mask"]
                local_num_valid_responses = batch["local_num_valid_responses"]
                prev_logprobs = batch["prev_logprobs"]
                advantages = batch["advantages"]
                tokens = batch["response_tokens"]
                reference_policy_logprobs = batch["reference_policy_logprobs"]
                generation_logprobs = batch["generation_logprobs"]

                # Agent multi-turn: override mask with agent_loss_mask
                if getattr(args, 'agent_multi_turn', False):
                    agent_loss_mask = batch.get("agent_loss_mask")
                    if agent_loss_mask is not None:
                        # Trim to match sequence dimension
                        if agent_loss_mask.size(-1) != mask.size(-1):
                            agent_loss_mask = agent_loss_mask[..., :mask.size(-1)]
                        mask = agent_loss_mask

                    # Incorporate process rewards into advantages
                    process_rewards = batch.get("process_rewards")
                    if process_rewards is not None:
                        if process_rewards.size(-1) != advantages.size(-1):
                            # process_rewards is per-token, advantages is per-sample
                            # Average process rewards over valid tokens for each sample
                            pr_masked = process_rewards[..., :mask.size(-1)] * mask
                            pr_per_sample = pr_masked.sum(dim=-1, keepdim=True) / (mask.sum(dim=-1, keepdim=True) + 1e-8)
                            advantages = advantages + args.process_reward_weight * pr_per_sample
                
                # generation_logprobs = torch.zeros_like(
                #     reference_policy_logprobs, dtype=torch.float32
                # )
                
                curr_logprobs = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=parallel_logits, target=tokens, higher_stability=True
                )
                # Ensure prompt/padding positions have safe importance weights (=1.0)
                generation_logprobs = torch.where(mask.bool(), generation_logprobs, prev_logprobs)
                # Token-level correction
                actor_importance_weights_expanded = torch.exp(
                    prev_logprobs - generation_logprobs
                ).detach()
                actor_importance_weights_expanded = torch.nan_to_num(
                    actor_importance_weights_expanded, nan=0.0, posinf=0.0, neginf=0.0
                )
                # TIS see https://fengyao.notion.site/off-policy-rl
                actor_importance_weights_expanded = torch.clamp(
                    actor_importance_weights_expanded,
                    max=2.0,
                )
                actor_importance_weights = actor_importance_weights_expanded
                del actor_importance_weights_expanded
                importance_weights_to_use = actor_importance_weights
                reference_policy_kl_penalty = args.kl_penalty
                # reference_policy_kl_penalty = 0.0
                kl = (
                    reference_policy_kl_penalty
                    * calculate_kl_penalty_joschu2020(
                        logprobs_policy=curr_logprobs,
                        logprobs_reference=reference_policy_logprobs,
                    )
                )
                # GRPO two-level normalization:
                # Inner: per-response token average (1/|o_i|)
                # Outer: per-group response average (1/G)
                per_response_len = mask.sum(dim=-1).clamp(min=1)  # (B,)

                kl_per_response = (kl * mask).sum(dim=-1) / per_response_len  # (B,)
                kl = kl_per_response.sum() / (local_num_valid_responses + 1e-8)

                ratios = (curr_logprobs - prev_logprobs).exp()
                ratio_clip_min, ratio_clip_max = 0.2, 0.28
                ratios_clamped = ratios.clamp(
                    1.0 - ratio_clip_min, 1.0 + ratio_clip_max
                )
                loss1 = -advantages * ratios
                loss2 = -advantages * ratios_clamped
                clip_loss = torch.max(loss1, loss2)

                per_response_clip_loss = (importance_weights_to_use * clip_loss * mask).sum(dim=-1) / per_response_len  # (B,)
                actor_loss = per_response_clip_loss.sum() / (local_num_valid_responses + 1e-8)
                
                loss = actor_loss + kl
                reduced_actor_loss = average_losses_across_data_parallel_group([loss])
                return (
                    loss,
                    {"loss": reduced_actor_loss,},
                )
    
            return parallel_logits, loss_func
    
        return fwd_output_and_loss_func
        
    def train_step(self,data_iterator):
        """Single training step."""
        args = get_args()
        timers = get_timers()
        
    
        # Set grad to zero.
        for partition in self.model:
            try:
                partition.zero_grad_buffer()
            except:
                partition.zero_grad_buffer(zero_buffer=(not args.use_distributed_optimizer))
        self.optimizer.zero_grad()
    
        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=self.get_actor_forward_output_and_loss_func(),
            data_iterator=data_iterator,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.max_padding_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)
    
        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()
    
        # Update parameters.
        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        timers('optimizer').stop()
    
        try:
            if update_successful:
                self.optimizer.gather_model_params(args, timers)
        except:
            pass
    
        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            self.opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1
    
        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()
    
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        return {}, skipped_iter, grad_norm, num_zeros_in_grad
        
   

    def compute_rollout_metrics(self, rollout_batch,is_valid=False):
        table = {}

        prompt_lengths = rollout_batch["prompt_lengths"]
        response_lengths = rollout_batch["response_lengths"]
        response_tokens = rollout_batch["response_tokens"]
        rewards = rollout_batch["rewards"]
        is_end = rollout_batch["is_end"]

        # take the first sample for logging
        reward = rewards[0]
        prompt_length = prompt_lengths[0]
        response_length = response_lengths[0]
        response_token = response_tokens[0]

        table["reward"] = reward.item()
        
        try:
            table["prompt"] = self.tokenizer.decode(response_token[:prompt_length].tolist())
            table["response"] = self.tokenizer.decode(response_token[prompt_length:response_length].tolist())
        except:
            table["response"] = response_token
        valid_name = "valid" if is_valid else "train"

        metrics = {
            "table": table,
            "train_or_valid": valid_name,
            "rollout_size": prompt_lengths.size(0),
            "avg_response_length": response_lengths.float().mean().item(),
            "avg_prompt_length": prompt_lengths.float().mean().item(),
            "avg_generation_length": (response_lengths - prompt_lengths).float().mean().item(),
            "max_generation_length": (response_lengths - prompt_lengths).float().max().item(),
            "min_generation_length": (response_lengths - prompt_lengths).float().min().item(),
            "avg_reward": rewards.mean().item(),
            "avg_fraction_of_samples_properly_ended": is_end.float().mean().item(),
        }

        return metrics

    @torch.no_grad()
    def evaluate(self):
        args = get_args()
        step = 0
        set_jit_fusion_options()
        for model_chunk in self.model:
            model_chunk.eval()
        avg_rewards = 0.0
        n = 0
        rollout_batches= []
        self.inference_engine.wake_up()
        self.convert()
        results = []
        for roll_iter in range(args.eval_iters):
            sampler_iter = iter(self.valid_dataloader.batch_sampler)
            rollout_num_microbatches = compute_num_rollout_microbatches(args, self.valid_dataloader)
            collate_fn = torch.utils.data.dataloader.default_collate
            batch_iterator = DefaultBatchIterator(sampler_iter, rollout_num_microbatches, self.valid_dataloader.dataset, collate_fn)
            prompt = []
            for batch in batch_iterator:
                # for _ in range(args.vllm_num_rollout_samples):
                token_ids = batch["text"].tolist()
                rollout_batch = {}
                rollout_batch["prompt_tokens"] = token_ids
                rollout_batch["prompt_lengths"] = [len(x) for x in token_ids]
                
                rollout_batch["ground_truth"] = batch["ground_truth"]
                rollout_batches.append(rollout_batch)
                for token_id in token_ids:
                    prompt.append(token_id)
            outputs = self.inference_engine.generate(
                    prompt_token_ids=prompt,  # because we have already convert it to prompt token id
                    sampling_params=SamplingParams(temperature=0.6, top_p=0.9, max_tokens=8192, logprobs=1),
                    use_tqdm=True,
                )
            
            results.extend(outputs)
            
        result_idx = 0
        for rollout_batch in rollout_batches:
            num_prompts = len(rollout_batch["prompt_tokens"])
            output = results[result_idx]
            rollout_batch["response_tokens"] = torch.LongTensor([output.prompt_token_ids + output.outputs[0].token_ids]).cpu()
            rollout_batch["response_lengths"] = torch.LongTensor([len(output.prompt_token_ids) + len(output.outputs[0].token_ids)]).cpu()
            reward = compute_score(output.outputs[0].text, str(rollout_batch["ground_truth"][0]))
            is_end = torch.LongTensor([1 if output.outputs[0].token_ids[-1]==self.tokenizer.eos_token_id else 0]).cpu()
            rollout_batch["is_end"] = is_end
            rollout_batch["rewards"] = torch.FloatTensor([reward]).cpu()
            rollout_batch.pop("ground_truth")
            result_idx += num_prompts

        self.inference_engine.sleep(2)
        torch._C._cuda_clearCublasWorkspaces()
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
            
        for rollout_batch in rollout_batches:
            rollout_batch["prompt_tokens"] = batch_pad_to_fixed_len(torch.LongTensor(rollout_batch["prompt_tokens"]), args.max_padding_length, self.tokenizer.eos_token_id)
            rollout_batch["prompt_lengths"] = broadcast_tensor_within_pp(torch.LongTensor(rollout_batch["prompt_lengths"]), from_last=False)
            rollout_batch["prompt_tokens"] = broadcast_tensor_within_pp(rollout_batch["prompt_tokens"], from_last=False)
            rollout_batch["prompt_lengths"] = broadcast_tensor_within_pp(rollout_batch["prompt_lengths"], from_last=False)
            rollout_batch["response_tokens"] = broadcast_tensor_within_pp(rollout_batch["response_tokens"], from_last=False)
            rollout_batch["response_lengths"] = broadcast_tensor_within_pp(rollout_batch["response_lengths"], from_last=False)
            rollout_batch["rewards"] = broadcast_tensor_within_pp(rollout_batch["rewards"], from_last=False)
            rollout_batch["is_end"] = broadcast_tensor_within_pp(rollout_batch["is_end"], from_last=False)
            rewards = rollout_batch["rewards"].mean().item()
            avg_rewards += rewards
            n+=1
            max_length = rollout_batch["response_lengths"].max().item()

            # Map pad_id to eos_id in case tokenizer does not have a pad_id
            rollout_batch["response_tokens"] = rollout_batch["response_tokens"][..., :max_length].contiguous()
            rollout_batch["response_tokens"] = broadcast_2d_tensor_within_mp(rollout_batch["response_tokens"], dtype=rollout_batch["response_tokens"].dtype)

        unbalanced_local_batch = ReinforceRolloutBatch.from_rollout_batches(
                rollout_batches,
                eos_id=self.tokenizer.eos_token_id,
                rollout_batch_seq_length=args.max_padding_length,
            )
        global_rollout_batch = unbalanced_local_batch.gather_and_balance_globally()

        padded_rollout_sequence_length = global_rollout_batch["response_tokens"].size(-1)

        balanced_local_batch = global_rollout_batch.chunk(
            rank=mpu.get_data_parallel_rank(),
            split_size=mpu.get_data_parallel_world_size(),
            seed=step,
        )
        step += 1
        print_rank_0(self.compute_rollout_metrics(balanced_local_batch,is_valid=True))
        avg_rewards = avg_rewards/n
        print_rank_0("step: "+str(self.step)+"avg_rewards: "+str(avg_rewards))
        for model_module in self.model:
            model_module.train()

    def save(self,step):
        args = get_args()
        save_checkpoint(step, self.model, None, None, args.num_floating_point_operations_so_far)

    def preparing_training_data(self):
        args = get_args()
        if getattr(args, 'agent_multi_turn', False):
            return self._preparing_training_data_multi_turn()
        return self._preparing_training_data_single_turn()

    def _preparing_training_data_multi_turn(self):
        """Multi-turn agent rollout path."""
        args = get_args()

        sampler_iter = iter(cyclic_iter(self.train_dataloader.batch_sampler))
        rollout_num_microbatches = compute_num_rollout_microbatches(args, self.train_dataloader)
        collate_fn = torch.utils.data.dataloader.default_collate
        batch_iterator = DefaultBatchIterator(
            sampler_iter, rollout_num_microbatches,
            self.train_dataloader.dataset, collate_fn,
        )

        # Collect prompts and ground truths
        all_prompt_tokens = []
        all_ground_truths = []
        for batch in batch_iterator:
            for _ in range(args.vllm_num_rollout_samples):
                token_ids = batch["text"].tolist()
                for tid in token_ids:
                    all_prompt_tokens.append(tid)
                    # ground_truth is a list from collate
                gt_list = batch["ground_truth"]
                for gt in (gt_list if isinstance(gt_list, list) else [gt_list]):
                    all_ground_truths.append(str(gt))

        # Run multi-turn rollout
        trajectories = self.multi_turn_orchestrator.rollout_batch(
            all_prompt_tokens, all_ground_truths,
        )

        # Compute rewards and build rollout batches
        rollout_batches = []
        for sample_idx, traj in enumerate(trajectories):
            # Determine reward source: bio env or standard rpf
            final_reward = 0.0
            has_bio_finalize = any(
                t.tool_call is not None and t.tool_call.tool_name == "workflow_finalize"
                for t in traj.turns
            )
            if has_bio_finalize and self.bio_tool_executor is not None:
                # Bio workflow reward from 6-layer verification
                env = self.bio_tool_executor.get_or_create_env(sample_idx)
                if env._finalized:
                    # Re-run finalize to get reward
                    env._finalized = False
                    _, bio_reward, _ = env.finalize()
                    final_reward = bio_reward
                else:
                    final_reward = 0.0
            else:
                # Standard reward (math/code tasks)
                last_text = traj.last_generated_text
                final_reward = compute_score(last_text, traj.ground_truth)

            # Agent reward (final + process)
            reward_info = self.agent_reward.compute_trajectory_reward(traj, final_reward)

            # Convert trajectory to training tensors
            tensors = traj.to_training_tensors(
                max_seq_length=args.max_padding_length,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Assign process rewards to tokens
            self.agent_reward.assign_process_rewards_to_tokens(
                traj,
                reward_info["per_turn_rewards"],
                tensors["process_rewards"],
                tensors["prompt_length"],
            )

            is_end = 1 if traj.is_complete else 0
            rollout_batch = {
                "prompt_tokens": torch.LongTensor(traj.original_prompt_tokens).unsqueeze(0),
                "prompt_lengths": torch.LongTensor([tensors["prompt_length"]]),
                "response_tokens": tensors["response_tokens"].unsqueeze(0),
                "response_lengths": torch.LongTensor([tensors["response_length"]]),
                "rewards": torch.FloatTensor([reward_info["total_reward"]]),
                "is_end": torch.LongTensor([is_end]),
                "generation_logprobs": tensors["generation_logprobs"].unsqueeze(0),
                "agent_loss_mask": tensors["agent_loss_mask"].unsqueeze(0),
                "process_rewards": tensors["process_rewards"].unsqueeze(0),
            }

            # Broadcast across pipeline stages
            rollout_batch["prompt_tokens"] = batch_pad_to_fixed_len(
                rollout_batch["prompt_tokens"], args.max_padding_length, self.tokenizer.eos_token_id
            )
            for key in rollout_batch:
                rollout_batch[key] = broadcast_tensor_within_pp(rollout_batch[key], from_last=False)

            max_length = rollout_batch["response_lengths"].max().item()
            rollout_batch["response_tokens"] = rollout_batch["response_tokens"][..., :max_length].contiguous()
            rollout_batch["response_tokens"] = broadcast_2d_tensor_within_mp(
                rollout_batch["response_tokens"], dtype=rollout_batch["response_tokens"].dtype
            )
            rollout_batch["generation_logprobs"] = broadcast_2d_tensor_within_mp(
                rollout_batch["generation_logprobs"], dtype=rollout_batch["generation_logprobs"].dtype
            )
            rollout_batch["agent_loss_mask"] = broadcast_2d_tensor_within_mp(
                rollout_batch["agent_loss_mask"], dtype=rollout_batch["agent_loss_mask"].dtype
            )
            rollout_batch["process_rewards"] = broadcast_2d_tensor_within_mp(
                rollout_batch["process_rewards"], dtype=rollout_batch["process_rewards"].dtype
            )
            rollout_batches.append(rollout_batch)

        self.inference_engine.sleep(2)
        torch._C._cuda_clearCublasWorkspaces()
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()

        unbalanced_local_batch = ReinforceRolloutBatch.from_rollout_batches(
            rollout_batches,
            eos_id=self.tokenizer.eos_token_id,
            rollout_batch_seq_length=args.max_padding_length,
        )
        global_rollout_batch = unbalanced_local_batch.gather_and_balance_globally()
        balanced_local_batch = global_rollout_batch.chunk(
            rank=mpu.get_data_parallel_rank(),
            split_size=mpu.get_data_parallel_world_size(),
            seed=self.step,
        )
        self.step += 1

        batched_response_tokens = balanced_local_batch["response_tokens"]
        self._memory_manager.onload_weights()
        rollout_logprobs = self.get_inference_log_probs(
            self.model, batched_response_tokens, self.tokenizer.eos_token_id, 8
        )
        balanced_local_batch["prev_logprobs"] = rollout_logprobs
        self._memory_manager.offload_weights()

        self.ref_memory_manager.onload_weights()
        rollout_ref_logprobs = self.get_inference_log_probs(
            self.ref_model, batched_response_tokens, self.tokenizer.eos_token_id, 8
        )
        balanced_local_batch["ref_logprobs"] = rollout_ref_logprobs
        self.ref_memory_manager.offload_weights()

        reinforce_rollout_data = {}
        prompt_lengths = balanced_local_batch["prompt_lengths"]
        response_lengths = balanced_local_batch["response_lengths"]
        prompt_tokens = balanced_local_batch["prompt_tokens"]
        response_tokens = balanced_local_batch["response_tokens"]
        rewards = balanced_local_batch["rewards"]
        logprobs = balanced_local_batch["prev_logprobs"]
        sample_mask = balanced_local_batch["is_end"]

        # Use agent_loss_mask instead of create_mask for multi-turn
        if "agent_loss_mask" in balanced_local_batch and balanced_local_batch["agent_loss_mask"] is not None:
            token_mask = balanced_local_batch["agent_loss_mask"]
            # Ensure shape matches logprobs
            if token_mask.size(-1) != logprobs.size(-1):
                token_mask = token_mask[..., :logprobs.size(-1)]
        else:
            token_mask = create_mask(values=logprobs, prompt_lengths=prompt_lengths, response_lengths=response_lengths)

        mask = token_mask * sample_mask.unsqueeze(-1)
        local_num_valid_responses = sample_mask.sum()
        local_num_valid_responses = broadcast_tensor_within_pp(local_num_valid_responses)

        baseline, std = calculate_baseline_and_std_per_prompt(
            prompt_tokens, rewards,
            torch.ones_like(rewards),
            leave_one_out_baseline=True,
        )
        advantages = (rewards - baseline).unsqueeze(-1)
        zero_std_mask = std > 0
        advantages[zero_std_mask] = (
            advantages[zero_std_mask] / std.unsqueeze(-1)[zero_std_mask]
        )

        reinforce_rollout_data["mask"] = mask
        reinforce_rollout_data["prev_logprobs"] = balanced_local_batch["prev_logprobs"]
        reinforce_rollout_data["advantages"] = advantages
        reinforce_rollout_data["response_tokens"] = response_tokens
        reinforce_rollout_data["local_num_valid_responses"] = local_num_valid_responses
        reinforce_rollout_data["reference_policy_logprobs"] = balanced_local_batch["ref_logprobs"]
        reinforce_rollout_data["generation_logprobs"] = balanced_local_batch["generation_logprobs"]

        # Pass agent-specific tensors
        if "agent_loss_mask" in balanced_local_batch:
            reinforce_rollout_data["agent_loss_mask"] = balanced_local_batch["agent_loss_mask"]
        if "process_rewards" in balanced_local_batch:
            reinforce_rollout_data["process_rewards"] = balanced_local_batch["process_rewards"]

        rollout_size = reinforce_rollout_data["response_tokens"].size(0)
        dp_size = mpu.get_data_parallel_world_size()
        num_to_load_on_each_dp = divide(args.global_batch_size, dp_size)
        rollout_dataloader_iter = get_iterator_k_split(
            reinforce_rollout_data, divide(rollout_size, num_to_load_on_each_dp)
        )
        return rollout_dataloader_iter

    def _preparing_training_data_single_turn(self):
        args = get_args()
        sampler_iter = iter(cyclic_iter(self.train_dataloader.batch_sampler))
        rollout_num_microbatches = compute_num_rollout_microbatches(args, self.train_dataloader)
        collate_fn = torch.utils.data.dataloader.default_collate
        batch_iterator = DefaultBatchIterator(sampler_iter, rollout_num_microbatches, self.train_dataloader.dataset, collate_fn)
        rollout_batches, results = [], []
        prompt = []
        for batch in batch_iterator:

            for _ in range(args.vllm_num_rollout_samples):
                token_ids = batch["text"].tolist()
                rollout_batch = {}
                
                rollout_batch["prompt_tokens"] = token_ids
                rollout_batch["prompt_lengths"] = [len(x) for x in token_ids]
                rollout_batch["ground_truth"] = batch["ground_truth"]
                rollout_batches.append(rollout_batch)
                for token_id in token_ids:
                    prompt.append(token_id)
        vllm_batch_size = 1024
        iter_num = len(prompt)//vllm_batch_size if len(prompt)%vllm_batch_size==0 else len(prompt)//vllm_batch_size+1

        for i in range(iter_num):
            outputs = self.inference_engine.generate(
                prompt_token_ids=prompt[i*vllm_batch_size:(i+1)*vllm_batch_size],  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            results.extend(outputs)
 
        cache_li = []
        for rollout_batch in rollout_batches:
            for i,output in enumerate(results):
                if i not in cache_li:
                    if "ground_truth" in rollout_batch:
                        if [output.prompt_token_ids] == rollout_batch["prompt_tokens"]:
                            logprobs_list = []
                            for token_id, logprob_obj in zip(
                                output.outputs[0].token_ids,
                                output.outputs[0].logprobs
                            ):
                                    if logprob_obj and token_id in logprob_obj:
                                        logprobs_list.append(logprob_obj[token_id].logprob)
                                    else:
                                        logprobs_list.append(0.0)
                            rollout_batch["generation_logprobs"] =  torch.FloatTensor([logprobs_list]).cpu()
                            rollout_batch["response_tokens"] = torch.LongTensor([output.prompt_token_ids + output.outputs[0].token_ids]).cpu()
                            rollout_batch["response_lengths"] = torch.LongTensor([len(output.prompt_token_ids) + len(output.outputs[0].token_ids)]).cpu()
                            reward = compute_score(output.outputs[0].text, str(rollout_batch["ground_truth"][0]))
                            is_end = torch.LongTensor([1 if output.outputs[0].token_ids[:args.max_padding_length][-1]==self.tokenizer.eos_token_id else 0]).cpu()

                            rollout_batch["is_end"] = is_end
                            rollout_batch["rewards"] = torch.FloatTensor([reward]).cpu()
                            rollout_batch.pop("ground_truth")
                            cache_li.append(i)

        self.inference_engine.sleep(2)
        torch._C._cuda_clearCublasWorkspaces()
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
        for rollout_batch in rollout_batches:
            rollout_batch["prompt_tokens"] = batch_pad_to_fixed_len(torch.LongTensor(rollout_batch["prompt_tokens"]), args.max_padding_length, self.tokenizer.eos_token_id)
            rollout_batch["prompt_lengths"] = broadcast_tensor_within_pp(torch.LongTensor(rollout_batch["prompt_lengths"]), from_last=False)
            rollout_batch["prompt_tokens"] = broadcast_tensor_within_pp(rollout_batch["prompt_tokens"], from_last=False)
            rollout_batch["prompt_lengths"] = broadcast_tensor_within_pp(rollout_batch["prompt_lengths"], from_last=False)
            rollout_batch["response_tokens"] = broadcast_tensor_within_pp(rollout_batch["response_tokens"], from_last=False)
            rollout_batch["response_lengths"] = broadcast_tensor_within_pp(rollout_batch["response_lengths"], from_last=False)
            rollout_batch["generation_logprobs"] = broadcast_tensor_within_pp(rollout_batch["generation_logprobs"], from_last=False)
            rollout_batch["rewards"] = broadcast_tensor_within_pp(rollout_batch["rewards"], from_last=False)
            rollout_batch["is_end"] = broadcast_tensor_within_pp(rollout_batch["is_end"], from_last=False)
            max_length = rollout_batch["response_lengths"].max().item()


            rollout_batch["response_tokens"] = rollout_batch["response_tokens"][..., :max_length].contiguous()
            rollout_batch["response_tokens"] = broadcast_2d_tensor_within_mp(rollout_batch["response_tokens"], dtype=rollout_batch["response_tokens"].dtype)
            rollout_batch["generation_logprobs"] = broadcast_2d_tensor_within_mp(rollout_batch["generation_logprobs"], dtype=rollout_batch["generation_logprobs"].dtype)


        unbalanced_local_batch = ReinforceRolloutBatch.from_rollout_batches(
                rollout_batches,
                eos_id=self.tokenizer.eos_token_id,
                rollout_batch_seq_length=args.max_padding_length,
            )
        global_rollout_batch = unbalanced_local_batch.gather_and_balance_globally()

        padded_rollout_sequence_length = global_rollout_batch["response_tokens"].size(-1)

        balanced_local_batch = global_rollout_batch.chunk(
            rank=mpu.get_data_parallel_rank(),
            split_size=mpu.get_data_parallel_world_size(),
            seed=self.step,
        )
        self.step += 1
        batched_response_tokens = balanced_local_batch["response_tokens"]
        self._memory_manager.onload_weights()
        rollout_logprobs = self.get_inference_log_probs(self.model,batched_response_tokens,self.tokenizer.eos_token_id,8)
        balanced_local_batch["prev_logprobs"] = rollout_logprobs
        self._memory_manager.offload_weights()
        self.ref_memory_manager.onload_weights()
        rollout_ref_logprobs = self.get_inference_log_probs(self.ref_model,batched_response_tokens,self.tokenizer.eos_token_id,8)
        balanced_local_batch["ref_logprobs"] = rollout_ref_logprobs
        self.ref_memory_manager.offload_weights()
        reinforce_rollout_data = {}
        prompt_lengths = balanced_local_batch["prompt_lengths"]
        response_lengths = balanced_local_batch["response_lengths"]
        prompt_tokens = balanced_local_batch["prompt_tokens"]
        response_tokens = balanced_local_batch["response_tokens"]
        rewards = balanced_local_batch["rewards"]
        logprobs = balanced_local_batch["prev_logprobs"]
        sample_mask = balanced_local_batch["is_end"]


        token_mask = create_mask(values=logprobs, prompt_lengths=prompt_lengths, response_lengths=response_lengths)
        mask = token_mask * sample_mask.unsqueeze(-1)
        local_num_valid_responses = sample_mask.sum()
        local_num_valid_responses = broadcast_tensor_within_pp(local_num_valid_responses)


        baseline, std = calculate_baseline_and_std_per_prompt(
                    prompt_tokens,
                    rewards,
                    torch.ones_like(rewards),
                    leave_one_out_baseline=True,
        )
        advantages = (rewards - baseline).unsqueeze(-1)

        zero_std_mask = std > 0
        advantages[zero_std_mask] = (
            advantages[zero_std_mask] / std.unsqueeze(-1)[zero_std_mask]
        )
        # collect everything we need to train GRPO
        reinforce_rollout_data["mask"] = mask
        reinforce_rollout_data["prev_logprobs"] = balanced_local_batch["prev_logprobs"]
        reinforce_rollout_data["advantages"] = advantages
        reinforce_rollout_data["response_tokens"] = response_tokens
        reinforce_rollout_data["local_num_valid_responses"] = local_num_valid_responses
        reinforce_rollout_data["reference_policy_logprobs"] = balanced_local_batch["ref_logprobs"]
        reinforce_rollout_data["generation_logprobs"] = balanced_local_batch["generation_logprobs"]

        rollout_size = reinforce_rollout_data["response_tokens"].size(0)
        
        dp_size = mpu.get_data_parallel_world_size()
        num_to_load_on_each_dp = divide(args.global_batch_size, dp_size)

        rollout_dataloader_iter = get_iterator_k_split(
            reinforce_rollout_data, divide(rollout_size, num_to_load_on_each_dp)
        )
        return rollout_dataloader_iter
    
    def train(self):
        args = get_args()

        set_jit_fusion_options()
        for roll_iter in range(args.train_iters):

            if args.eval_interval and self.step % args.eval_interval == 0 and \
                args.do_valid:

                print_rank_0("==================start valid============")
                self.evaluate()
                print_rank_0("==================end valid============")


            self.inference_engine.wake_up()
            self.convert()
            rollout_dataloader_iter = self.preparing_training_data()
            self.inference_engine.sleep(2)
            self._memory_manager.onloads()

            for batch in rollout_dataloader_iter:
                sequence_length = batch["response_tokens"].size(1)
                eos_id = self.tokenizer.eos_token_id
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(batch["response_tokens"], eos_id, False, False, False)
                batch["attention_mask"] = attention_mask.expand(batch["response_tokens"].size(0), -1, -1, -1)
                batch["position_ids"] = position_ids.expand(batch["response_tokens"].size(0), -1)

                data_iter = get_iterator_k_split(batch, get_num_microbatches())
                num_microbatches = get_num_microbatches()

                if args.profile and torch.distributed.get_rank() in args.profile_ranks:
                    if args.use_pytorch_profiler:
                        prof.step()
                    elif self.step == args.profile_step_start:
                        torch.cuda.cudart().cudaProfilerStart()
                        torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

                ft_integration.on_checkpointing_start()
                maybe_finalize_async_save(blocking=False)
                ft_integration.on_checkpointing_end(is_async_finalization=True)

                # Update number of microbatches first without consistency check to decide if a
                # checkpoint should be saved. If the number of microbatches is different
                # from the previous iteration, save a checkpoint. Then run consistency check
                # to make sure training configuration is still valid.
                update_num_microbatches(args.consumed_train_samples, consistency_check=False, verbose=True)
                if get_num_microbatches() != num_microbatches and self.step != 0:
                    assert get_num_microbatches() > num_microbatches, \
                        (f"Number of microbatches should be increasing due to batch size rampup; "
                         f"instead going from {num_microbatches} to {get_num_microbatches()}")

                # Run training step.
                args.curr_iteration = self.step
                ft_integration.on_training_step_start()
                loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad = self.train_step(data_iter)
                ft_integration.on_training_step_end()
                batch_size = mpu.get_data_parallel_world_size() * \
                     args.micro_batch_size * \
                     get_num_microbatches()
                args.consumed_train_samples += batch_size
                # Logging.
                if not self.optimizer.is_stub_optimizer:
                    loss_scale = self.optimizer.get_loss_scale().item()
                else:
                    loss_scale = 1.0
                params_norm = None

                if args.log_params_norm:
                    params_norm = calc_params_l2_norm(self.model)
                learning_rate = None
                decoupled_learning_rate = None
                for param_group in self.optimizer.param_groups:
                    if param_group['is_decoupled_lr']:
                        decoupled_learning_rate = param_group['lr']
                    else:
                        learning_rate = param_group['lr']
                print_rank_0(f"step: {self.step} | loss: {loss_reduced} | grad_norm: {grad_norm} | lr: {learning_rate} | loss_scale: {loss_scale} | skipped: {skipped_iter}")

            if args.save and self.step != 0 and self.step % args.save_interval == 0:
                print_rank_0("==================start saving checkpoint============")
                self.save(self.step)
                print_rank_0("==================end saving checkpoint============")
            self._memory_manager.offloads()

if __name__ == "__main__":
    
    train_valid_test_datasets_provider.is_distributed = True
    trainer = Trainer()
    trainer.train()
