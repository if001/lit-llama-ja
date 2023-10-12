# original
# https://github.com/Lightning-AI/lit-gpt/blob/main/scripts/convert_lit_checkpoint.py

import gc
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

import torch
from lightning.fabric.utilities.load import _NotYetLoadedTensor as NotYetLoadedTensor

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# from lit_gpt import Config
from lit_llama.config_llama2 import Llama2Config
# from lit_gpt.utils import incremental_save, lazy_load
from lit_llama.utils import incremental_save
from lightning.fabric.utilities.load import _lazy_load as lazy_load
# from scripts.convert_hf_checkpoint import layer_template, load_param

def layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    split = layer_name.split(".")
    number = int(split[idx])
    split[idx] = "{}"
    from_name = ".".join(split)
    return from_name, number

def load_param(param: Union[torch.Tensor, NotYetLoadedTensor], name: str, dtype: Optional[torch.dtype]) -> torch.Tensor:
    if hasattr(param, "_load_tensor"):
        # support tensors loaded via `lazy_load()`
        print(f"Loading {name!r} into RAM")
        param = param._load_tensor()
    if dtype is not None and type(dtype) is not NotYetLoadedTensor and dtype != param.dtype:
        print(f"Converting {name!r} from {param.dtype} to {dtype}")
        param = param.to(dtype)
    return param


def copy_weights_falcon(
    model_name: str,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
) -> None:
    weight_map = {
        "transformer.wte.weight": "transformer.word_embeddings.weight",
        "transformer.h.{}.attn.attn.weight": "transformer.h.{}.self_attention.query_key_value.weight",
        "transformer.h.{}.attn.proj.weight": "transformer.h.{}.self_attention.dense.weight",
        "transformer.h.{}.mlp.fc.weight": "transformer.h.{}.mlp.dense_h_to_4h.weight",
        "transformer.h.{}.mlp.proj.weight": "transformer.h.{}.mlp.dense_4h_to_h.weight",
        "transformer.ln_f.bias": "transformer.ln_f.bias",
        "transformer.ln_f.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }
    # the original model definition is different for each size
    if "7b" in model_name:
        weight_map.update(
            {
                "transformer.h.{}.norm_1.bias": "transformer.h.{}.input_layernorm.bias",
                "transformer.h.{}.norm_1.weight": "transformer.h.{}.input_layernorm.weight",
            }
        )
    elif "40b" in model_name or "180B" in model_name:
        weight_map.update(
            {
                "transformer.h.{}.norm_1.bias": "transformer.h.{}.ln_attn.bias",
                "transformer.h.{}.norm_1.weight": "transformer.h.{}.ln_attn.weight",
                "transformer.h.{}.norm_2.bias": "transformer.h.{}.ln_mlp.bias",
                "transformer.h.{}.norm_2.weight": "transformer.h.{}.ln_mlp.weight",
            }
        )
    else:
        raise NotImplementedError

    for name, param in lit_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, None)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_gpt_neox(
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
) -> None:
    weight_map = {
        "transformer.wte.weight": "gpt_neox.embed_in.weight",
        "transformer.h.{}.norm_1.bias": "gpt_neox.layers.{}.input_layernorm.bias",
        "transformer.h.{}.norm_1.weight": "gpt_neox.layers.{}.input_layernorm.weight",
        "transformer.h.{}.attn.attn.bias": "gpt_neox.layers.{}.attention.query_key_value.bias",
        "transformer.h.{}.attn.attn.weight": "gpt_neox.layers.{}.attention.query_key_value.weight",
        "transformer.h.{}.attn.proj.bias": "gpt_neox.layers.{}.attention.dense.bias",
        "transformer.h.{}.attn.proj.weight": "gpt_neox.layers.{}.attention.dense.weight",
        "transformer.h.{}.norm_2.bias": "gpt_neox.layers.{}.post_attention_layernorm.bias",
        "transformer.h.{}.norm_2.weight": "gpt_neox.layers.{}.post_attention_layernorm.weight",
        "transformer.h.{}.mlp.fc.bias": "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias",
        "transformer.h.{}.mlp.fc.weight": "gpt_neox.layers.{}.mlp.dense_h_to_4h.weight",
        "transformer.h.{}.mlp.proj.bias": "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias",
        "transformer.h.{}.mlp.proj.weight": "gpt_neox.layers.{}.mlp.dense_4h_to_h.weight",
        "transformer.ln_f.bias": "gpt_neox.final_layer_norm.bias",
        "transformer.ln_f.weight": "gpt_neox.final_layer_norm.weight",
        "lm_head.weight": "embed_out.weight",
    }

    for name, param in lit_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, None)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def copy_weights_llama(
    config: Llama2Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
):
    weight_map = {
        "transformer.wte.weight": "model.embed_tokens.weight",
        "transformer.h.{}.norm_1.weight": "model.layers.{}.input_layernorm.weight",
        "transformer.h.{}.attn.proj.weight": "model.layers.{}.self_attn.o_proj.weight",
        "transformer.h.{}.norm_2.weight": "model.layers.{}.post_attention_layernorm.weight",
        "transformer.h.{}.mlp.fc_1.weight": "model.layers.{}.mlp.gate_proj.weight",
        "transformer.h.{}.mlp.fc_2.weight": "model.layers.{}.mlp.up_proj.weight",
        "transformer.h.{}.mlp.proj.weight": "model.layers.{}.mlp.down_proj.weight",
        "transformer.ln_f.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for name, param in lit_weights.items():
        if name.endswith(".attn.attn.weight"):
            from_name, number = layer_template(name, 2)
            q = "model.layers.{}.self_attn.q_proj.weight".format(number)
            k = "model.layers.{}.self_attn.k_proj.weight".format(number)
            v = "model.layers.{}.self_attn.v_proj.weight".format(number)
            qkv = load_param(param, name, None)
            qp, kp, vp = qkv_split(qkv, config)
            for to_name, param in zip((q, k, v), (qp, kp, vp)):
                if saver is not None:
                    param = saver.store_early(param)
                state_dict[to_name] = param
        else:
            if "transformer.h" in name and not name.endswith(".scale"):
                print('debug', name)
                from_name, number = layer_template(name, 2)
                print('debug from_name', from_name)
                to_name = weight_map[from_name]
                to_name = to_name.format(number)
            elif name.endswith(".scale"):
                print('scale', name)                
            else:
                to_name = weight_map[name]
            param = load_param(param, name, None)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param


def copy_weights_llama_v2(
    config: Llama2Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
):
    weight_map = {
        "transformer.wte.weight": "embed_tokens.weight",
        "transformer.h.{}.norm_1.scale": "layers.{}.input_layernorm.weight",
        "transformer.h.{}.attn.proj.weight": "layers.{}.self_attn.o_proj.weight",
        "transformer.h.{}.norm_2.scale": "layers.{}.post_attention_layernorm.weight",
        "transformer.h.{}.mlp.fc_1.weight": "layers.{}.mlp.gate_proj.weight",
        "transformer.h.{}.mlp.fc_2.weight": "layers.{}.mlp.up_proj.weight",
        "transformer.h.{}.mlp.proj.weight": "layers.{}.mlp.down_proj.weight",
        "transformer.ln_f.scale": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for name, param in lit_weights.items():        
        if "transformer.wte.weight" in name:
            print(name, param.size())
        if name.endswith(".attn.attn.weight"):
            from_name, number = layer_template(name, 2)
            q = "layers.{}.self_attn.q_proj.weight".format(number)
            k = "layers.{}.self_attn.k_proj.weight".format(number)
            v = "layers.{}.self_attn.v_proj.weight".format(number)
            qkv = load_param(param, name, None)
            qp, kp, vp = qkv_split(qkv, config)
            for to_name, param in zip((q, k, v), (qp, kp, vp)):
                print('debug: ', to_name, param.size())
                if saver is not None:
                    param = saver.store_early(param)
                state_dict[to_name] = param
        else:            
            if "transformer.h" in name:
                from_name, number = layer_template(name, 2)
                to_name = weight_map[from_name]
                to_name = to_name.format(number)
            # elif name.endswith(".scale"):
            #     print('scale', name)
            else:
                to_name = weight_map[name]
            param = load_param(param, name, None)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param


def copy_weights_hf_llama_v3(
    config: Llama2Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{}.mlp.fc_1.weight",
        "model.layers.{}.mlp.up_proj.weight": "transformer.h.{}.mlp.fc_2.weight",
        "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.proj.weight",
        "model.norm.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for name, param in hf_weights.items():
        if "model.layers" in name:
            from_name, number = layer_template(name, 2)
            qkv = qkv_weights.setdefault(number, [None, None, None])
            if "q_proj" in name:
                qkv[0] = param
            elif "k_proj" in name:
                qkv[1] = param
            elif "v_proj" in name:
                qkv[2] = param
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

    for i, (q, k, v) in list(qkv_weights.items()):
        if q is None or k is None or v is None:
            # split across different .bin files
            continue
        q = load_param(q, f"layer {i} q", dtype)
        k = load_param(k, f"layer {i} k", dtype)
        v = load_param(v, f"layer {i} v", dtype)
        q_per_kv = config.n_head // config.n_query_groups
        qs = torch.split(q, config.head_size * q_per_kv)
        ks = torch.split(k, config.head_size)
        vs = torch.split(v, config.head_size)
        cycled = [t for group in zip(qs, ks, vs) for t in group]
        qkv = torch.cat(cycled)
        state_dict[f"transformer.h.{i}.attn.attn.weight"] = qkv
        del qkv_weights[i]

def copy_weights_phi(
    config: Llama2Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
) -> None:
    weight_map = {
        "transformer.wte.weight": "layers.0.wte.weight",
        "transformer.h.{}.norm_1.bias": "layers.{}.ln.bias",
        "transformer.h.{}.norm_1.weight": "layers.{}.ln.weight",
        "transformer.h.{}.attn.attn.bias": "layers.{}.mixer.Wqkv.bias",
        "transformer.h.{}.attn.attn.weight": "layers.{}.mixer.Wqkv.weight",
        "transformer.h.{}.attn.proj.bias": "layers.{}.mixer.out_proj.bias",
        "transformer.h.{}.attn.proj.weight": "layers.{}.mixer.out_proj.weight",
        "transformer.h.{}.mlp.fc.bias": "layers.{}.mlp.fc1.bias",
        "transformer.h.{}.mlp.fc.weight": "layers.{}.mlp.fc1.weight",
        "transformer.h.{}.mlp.proj.bias": "layers.{}.mlp.fc2.bias",
        "transformer.h.{}.mlp.proj.weight": "layers.{}.mlp.fc2.weight",
        "transformer.ln_f.bias": f"layers.{config.n_layer + 1}.ln.bias",
        "transformer.ln_f.weight": f"layers.{config.n_layer + 1}.ln.weight",
        "lm_head.weight": f"layers.{config.n_layer + 1}.linear.weight",
        "lm_head.bias": f"layers.{config.n_layer + 1}.linear.bias",
    }

    for name, param in lit_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name]
            to_name = to_name.format(number + 1)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, None)
        if "attn.attn." in name:
            param = torch.cat(qkv_split(param, config))
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param


def qkv_split(
    param: Union[torch.Tensor, NotYetLoadedTensor], config: Llama2Config
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_per_kv = config.n_head // config.n_query_groups
    qs = []
    ks = []
    vs = []
    for chunk in torch.chunk(param, config.n_query_groups):
        split = torch.split(chunk, [config.head_size * q_per_kv, config.head_size, config.head_size])
        qs.append(split[0])
        ks.append(split[1])
        vs.append(split[2])
    q = torch.cat(qs)    
    k = torch.cat(ks)
    v = torch.cat(vs)
    return q, k, v


def check_conversion_supported(lit_weights: Dict[str, torch.Tensor]) -> None:
    if any("lora" in wn for wn in lit_weights):
        raise ValueError("Checkpoints with LoRA weights cannot be converted. Call `scripts/merge_lora.py` first.")
    if any("adapter" in wn or "gating_factor" in wn for wn in lit_weights):
        raise NotImplementedError("Converting adapter models is supported.")


@torch.inference_mode()
def convert_lit_checkpoint(model_size: str, output_path: Path, checkpoint_path: Path) -> None:
    # config = Llama2Config.from_json(config_path)
    config = Llama2Config.from_name(model_size)

    if "falcon" in config.name:
        copy_fn = partial(copy_weights_falcon, config.name)
    elif config._mlp_class == "LLaMAMLP":
        ## copy_fn = partial(copy_weights_llama, config)
        copy_fn = partial(copy_weights_llama_v2, config)
        ## qkv_weights = {}
        ## copy_fn = partial(copy_weights_hf_llama_v3, config, qkv_weights)
    else:
        copy_fn = copy_weights_gpt_neox

    # initialize a new empty state dict to hold our new weights
    sd = {}
    with incremental_save(output_path) as saver:
        lit_weights = lazy_load(checkpoint_path)
        lit_weights = lit_weights.get("model", lit_weights)
        check_conversion_supported(lit_weights)
        copy_fn(sd, lit_weights, saver=saver)
        gc.collect()
        saver.save(sd)
        
        print("="*100)
        for k in sd:
            print(k)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_lit_checkpoint, as_positional=False)