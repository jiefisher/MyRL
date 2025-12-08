import torch
from megatron.core.transformer import TransformerConfig
from transformers import PretrainedConfig


class McoreToHFWeightConverterBase:
    def __init__(self, hf_config: PretrainedConfig, mcore_config: TransformerConfig):
        self.hf_config = hf_config
        self.mcore_config = mcore_config

    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class McoreToHFWeightConverterDense(McoreToHFWeightConverterBase):
    def _convert_attention_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # 'decoder.layers.0.self_attention.linear_proj.weight'
        # 'decoder.layers.0.self_attention.linear_qkv.layer_norm_weight'
        # 'decoder.layers.0.self_attention.linear_qkv.weight'
        # 'decoder.layers.0.self_attention.linear_qkv.bias'
        layer_number = name.split(".")[2]
        convert_names = []
        if "self_attention.linear_qkv.bias" in name or "self_attention.linear_qkv.weight" in name:
            param_type = name.split(".")[-1]
            assert param_type == "bias" or param_type == "weight"
            convert_names.append(f"model.layers.{layer_number}.self_attn.q_proj.{param_type}")
            convert_names.append(f"model.layers.{layer_number}.self_attn.k_proj.{param_type}")
            convert_names.append(f"model.layers.{layer_number}.self_attn.v_proj.{param_type}")
            assert len(params) == 3
        elif "self_attention.linear_proj.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.self_attn.o_proj.weight")
            assert len(params) == 1
        elif "self_attention.linear_qkv.layer_norm_weight" in name:
            convert_names.append(f"model.layers.{layer_number}.input_layernorm.weight")
            assert len(params) == 1
        elif "self_attention.q_layernorm.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.self_attn.q_norm.weight")
            assert len(params) == 1
        elif "self_attention.k_layernorm.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.self_attn.k_norm.weight")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params

    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        # 'decoder.layers.0.mlp.linear_fc1.layer_norm_weight'
        # 'decoder.layers.0.mlp.linear_fc1.weight'
        # 'decoder.layers.0.mlp.linear_fc2.weight'
        layer_number = name.split(".")[2]
        convert_names = []
        if "mlp.linear_fc1.weight" in name:
            # split gate_proj and up_proj
            convert_names.append(f"model.layers.{layer_number}.mlp.gate_proj.weight")
            convert_names.append(f"model.layers.{layer_number}.mlp.up_proj.weight")
            assert len(params) == 2
        elif "mlp.linear_fc1.layer_norm_weight" in name:
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
            assert len(params) == 1
        elif "mlp.linear_fc2.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.down_proj.weight")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params

    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]], [params_one_group[0]]

        if "self_attention" in name:
            return self._convert_attention_param(name, params_one_group)
        elif "mlp" in name:
            return self._convert_mlp_param(name, params_one_group)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")