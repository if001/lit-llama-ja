# original
# https://github.com/Lightning-AI/lit-gpt/blob/main/scripts/convert_lit_checkpoint.py
"""
n_heads=[1, 1, 1, 4, 4, 4, 8, 8, 8, 8]
のようにhead sizeがlayerごとに異なるモデル用
"""


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


def copy_weights_llama_v2(
    config: Llama2Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
):
    weight_map = {
        "transformer.wte.weight": "model.embed_tokens.weight",
        "transformer.h.{}.norm_1.scale": "model.layers.{}.input_layernorm.weight",
        "transformer.h.{}.attn.proj.weight": "model.layers.{}.self_attn.o_proj.weight",
        "transformer.h.{}.norm_2.scale": "model.layers.{}.post_attention_layernorm.weight",
        "transformer.h.{}.mlp.fc_1.weight": "model.layers.{}.mlp.gate_proj.weight",
        "transformer.h.{}.mlp.fc_2.weight": "model.layers.{}.mlp.up_proj.weight",
        "transformer.h.{}.mlp.proj.weight": "model.layers.{}.mlp.down_proj.weight",
        "transformer.ln_f.scale": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for name, param in lit_weights.items():
        if name.endswith(".attn.attn.weight"):
            from_name, number = layer_template(name, 2)
            q = "model.layers.{}.self_attn.q_proj.weight".format(number)
            k = "model.layers.{}.self_attn.k_proj.weight".format(number)
            v = "model.layers.{}.self_attn.v_proj.weight".format(number)
            print(q)
            qkv = load_param(param, name, None)
            qp, kp, vp = qkv_split(qkv, config, number)
            for to_name, param in zip((q, k, v), (qp, kp, vp)):
                ## print('debug: ', to_name, param.size())
                if saver is not None:
                    param = saver.store_early(param)
                state_dict[to_name] = param
        else:
            if "transformer.h" in name:
                from_name, number = layer_template(name, 2)
                to_name = weight_map[from_name]
                to_name = to_name.format(number)
            else:
                to_name = weight_map[name]
            param = load_param(param, name, None)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param


def qkv_split(
    param: Union[torch.Tensor, NotYetLoadedTensor], 
    config: Llama2Config,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_per_kv = config.n_heads[layer_idx] // config.n_query_groups
    head_size = config.head_sizes[layer_idx]
    qs = []
    ks = []
    vs = []
    for chunk in torch.chunk(param, config.n_query_groups):
        split = torch.split(chunk, [head_size * q_per_kv, head_size, head_size])
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
    copy_fn = partial(copy_weights_llama_v2, config)

    # initialize a new empty state dict to hold our new weights
    sd = {}
    with incremental_save(output_path) as saver:
        lit_weights = lazy_load(checkpoint_path)
        lit_weights = lit_weights.get("model", lit_weights)
        check_conversion_supported(lit_weights)
        copy_fn(sd, lit_weights, saver=saver)
        gc.collect()
        saver.save(sd)
    print(f'saved...{output_path}')
    print("="*100)
    for k in sd:
        print(k)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_lit_checkpoint, as_positional=False)