import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union, List

import torch
from typing_extensions import Self

# import lit_gpt.model
# import lit_llama.model_llama2
from lit_llama.utils import find_multiple


@dataclass
class Llama2Config:
    org: str = "Lightning-AI"
    name: str = "lit-GPT"
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    # n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    lm_head_bias: bool = False
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"] = "GptNeoxMLP"
    gelu_approximate: str = "none"
    intermediate_size: Optional[int] = None
    rope_condense_ratio: int = 1
    rope_base: int = 10000
    nef: bool = False
    _description: str = ""
    
    _n_embd: List[int] = field(default_factory=list)
    _head_sizes: List[int] = field(default_factory=list)
    _intermediate_sizes: List[int] = field(default_factory=list)
    _rope_n_elems: List[int] = field(default_factory=list)
    # n_heads: List[int] = field(default_factory=list)    
    # n_query_groups_list: List[int] = field(default_factory=list)

    def __post_init__(self):
        # assert self.n_embd % self.n_head == 0
        # self.head_size = self.n_embd // self.n_head        

        for v in self._n_embd:
            self._head_sizes.append(v)

        self.n_layer = len(self._n_embd)

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        # if self.intermediate_size is None:
        #     if self._mlp_class == "LLaMAMLP":
        #         raise ValueError("The config needs to set the `intermediate_size`")
        #     self.intermediate_size = 4 * self.n_embd

        for v in self._n_embd:
            self._intermediate_sizes.append(4*v)

        for v in self._head_sizes:
            self._rope_n_elems.append(int(self.rotary_percentage * v))

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        if "condense_ratio" in kwargs:  # legacy name
            kwargs["rope_condense_ratio"] = kwargs.pop("condense_ratio")
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @classmethod
    def from_json(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        with open(path, encoding="utf-8") as fp:
            json_kwargs = json.load(fp)
        if "condense_ratio" in json_kwargs:  # legacy name
            json_kwargs["rope_condense_ratio"] = json_kwargs.pop("condense_ratio")
        if "condense_ratio" in kwargs:  # legacy name
            kwargs["rope_condense_ratio"] = kwargs.pop("condense_ratio")
        json_kwargs.update(kwargs)
        return cls(**json_kwargs)

    @property
    def mlp_class(self) -> Type:
        import lit_llama.model_llama2
        # `self._mlp_class` cannot be the type to keep the config json serializable
        return getattr(lit_llama.model_llama2, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from lit_llama.model import RMSNorm

            return RMSNorm
        return getattr(torch.nn, self._norm_class)

    def debug(self):
        print('='*100)
        print('print config...')
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print('='*100)

    def save(self, output_dir):
        """
        Save member variables of this instance to a JSON file.

        Parameters:
        - output_dir: The output dir of the JSON file to save to.
        """        
        member_vars = {k: v for k, v in self.__dict__.items() if not callable(v)}
        
        output_file = f'{output_dir}/model_config.json'
        with open(output_file, 'w') as f:
            json.dump(member_vars, f, ensure_ascii=False, indent=4)
        print(f'save model config... {output_file}')

########################
# Stability AI StableLM
########################
configs = [
    # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json
    dict(org="stabilityai", name="stablelm-base-alpha-3b"),
    # https://huggingface.co/stabilityai/stablelm-base-alpha-7b/blob/main/config.json
    dict(org="stabilityai", name="stablelm-base-alpha-7b", n_head=48, n_embd=6144, padding_multiple=256),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b/blob/main/config.json
    dict(org="stabilityai", name="stablelm-tuned-alpha-3b", n_head=32),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b/blob/main/config.json
    dict(org="stabilityai", name="stablelm-tuned-alpha-7b", n_head=48, n_embd=6144, padding_multiple=256),
]

####################
# EleutherAI Pythia
####################
pythia = [
    # https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json
    dict(org="EleutherAI", name="pythia-70m", block_size=2048, n_layer=6, n_embd=512, n_head=8, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-160m/blob/main/config.json
    dict(
        org="EleutherAI", name="pythia-160m", block_size=2048, n_layer=12, n_embd=768, n_head=12, padding_multiple=128
    ),
    # https://huggingface.co/EleutherAI/pythia-410m/blob/main/config.json
    dict(
        org="EleutherAI", name="pythia-410m", block_size=2048, n_layer=24, n_embd=1024, n_head=16, padding_multiple=128
    ),
    # https://huggingface.co/EleutherAI/pythia-1b/blob/main/config.json
    dict(org="EleutherAI", name="pythia-1b", block_size=2048, n_embd=2048, n_head=8, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-1.4b/blob/main/config.json
    dict(
        org="EleutherAI", name="pythia-1.4b", block_size=2048, n_layer=24, n_embd=2048, n_head=16, padding_multiple=128
    ),
    # https://huggingface.co/EleutherAI/pythia-2.8b/blob/main/config.json
    dict(org="EleutherAI", name="pythia-2.8b", block_size=2048, n_layer=32, n_embd=2560, padding_multiple=128),
    # https://huggingface.co/EleutherAI/pythia-6.9b/blob/main/config.json
    dict(org="EleutherAI", name="pythia-6.9b", block_size=2048, n_layer=32, padding_multiple=256),
    # https://huggingface.co/EleutherAI/pythia-12b/blob/main/config.json
    dict(org="EleutherAI", name="pythia-12b", block_size=2048, n_layer=36, n_embd=5120, n_head=40),
]
configs.extend(pythia)
for c in pythia:
    copy = c.copy()
    copy["name"] = f"{c['name']}-deduped"
    configs.append(copy)


####################################
# togethercomputer RedPajama INCITE
####################################
redpajama_incite = [
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1/blob/main/config.json
    dict(
        org="togethercomputer",
        name="RedPajama-INCITE-{}-3B-v1",
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base/blob/main/config.json
    dict(
        org="togethercomputer",
        name="RedPajama-INCITE-7B-{}",
        block_size=2048,
        n_layer=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
    # this redirects to the checkpoint above. kept for those who had the old weights already downloaded
    dict(
        org="togethercomputer",
        name="RedPajama-INCITE-{}-7B-v0.1",
        block_size=2048,
        n_layer=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
]
for c in redpajama_incite:
    for kind in ("Base", "Chat", "Instruct"):
        copy = c.copy()
        copy["name"] = c["name"].format(kind)
        configs.append(copy)


#################
# TII UAE Falcon
#################
falcon = [
    # https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json
    dict(
        org="tiiuae",
        name="falcon-7b{}",
        block_size=2048,
        vocab_size=65024,
        padded_vocab_size=65024,
        n_layer=32,
        n_head=71,
        n_embd=4544,
        rotary_percentage=1.0,
        n_query_groups=1,
        bias=False,
        # this is not in the config, but in the original model implementation, only for this config
        shared_attention_norm=True,
    ),
    # https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json
    dict(
        org="tiiuae",
        name="falcon-40b{}",
        block_size=2048,
        vocab_size=65024,
        padded_vocab_size=65024,
        n_layer=60,
        n_head=128,
        n_embd=8192,
        rotary_percentage=1.0,
        n_query_groups=8,
        bias=False,
    ),
]
for c in falcon:
    for kind in ("", "-instruct"):
        copy = c.copy()
        copy["name"] = c["name"].format(kind)
        configs.append(copy)

# https://huggingface.co/tiiuae/falcon-180b/blob/main/config.json
falcon180b = dict(
    org="tiiuae",
    name="falcon-180B{}",
    block_size=2048,
    vocab_size=65024,
    padded_vocab_size=65024,
    n_layer=80,
    n_head=232,
    n_embd=14848,
    rotary_percentage=1.0,
    n_query_groups=8,
    bias=False,
)

for kind in ("", "-chat"):
    copy = falcon180b.copy()
    copy["name"] = falcon180b["name"].format(kind)
    configs.append(copy)


#############################
# OpenLM Research Open LLaMA
#############################
open_LLaMA = [
    dict(
        org="openlm-research",
        name="open_llama_130M",
        vocab_size=35000,
        padding_multiple=64,
        block_size=2048,
        n_layer=12,
        n_head=10,
        n_embd=780,
        n_query_groups=10,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=1024,
        norm_eps=1.0e-6,
    ),
    # https://huggingface.co/openlm-research/open_llama_3b/blob/main/config.json
    dict(
        org="openlm-research",
        name="open_llama_3b",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=26,
        n_embd=3200,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=8640,
    ),
    # https://huggingface.co/openlm-research/open_llama_7b/blob/main/config.json
    dict(
        org="openlm-research",
        name="open_llama_7b",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/openlm-research/open_llama_13b/blob/main/config.json
    dict(
        org="openlm-research",
        name="open_llama_13b",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
]
configs.extend(open_LLaMA)


###############
# LMSYS Vicuna
###############
vicuna = [
    # https://huggingface.co/lmsys/vicuna-7b-v1.3/blob/main/config.json
    dict(
        org="lmsys",
        name="vicuna-7b-v1.3",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/lmsys/vicuna-13b-v1.3/blob/main/config.json
    dict(
        org="lmsys",
        name="vicuna-13b-v1.3",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/lmsys/vicuna-33b-v1.3/blob/main/config.json
    dict(
        org="lmsys",
        name="vicuna-33b-v1.3",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=60,
        n_head=52,
        n_embd=6656,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=17920,
    ),
    # https://huggingface.co/lmsys/vicuna-7b-v1.5/blob/main/config.json
    dict(
        org="lmsys",
        name="vicuna-7b-v1.5",
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/lmsys/vicuna-7b-v1.5-16k/blob/main/config.json
    dict(
        org="lmsys",
        name="vicuna-7b-v1.5-16k",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_condense_ratio=4,
    ),
    # https://huggingface.co/lmsys/vicuna-13b-v1.5/blob/main/config.json
    dict(
        org="lmsys",
        name="vicuna-13b-v1.5",
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/lmsys/vicuna-13b-v1.5-16k/blob/main/config.json
    dict(
        org="lmsys",
        name="vicuna-13b-v1.5-16k",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        rope_condense_ratio=4,
    ),
]
configs.extend(vicuna)


#################
# LMSYS LongChat
#################
long_chat = [
    # https://huggingface.co/lmsys/longchat-7b-16k/blob/main/config.json
    dict(
        org="lmsys",
        name="longchat-7b-16k",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_condense_ratio=8,
    ),
    # https://huggingface.co/lmsys/longchat-13b-16k/blob/main/config.json
    dict(
        org="lmsys",
        name="longchat-13b-16k",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        rope_condense_ratio=8,
    ),
]
configs.extend(long_chat)


######################
# NousResearch Hermes
######################
nous_research = [
    # https://huggingface.co/NousResearch/Nous-Hermes-llama-2-7b/blob/main/config.json
    dict(
        org="NousResearch",
        name="Nous-Hermes-llama-2-7b",
        padded_vocab_size=32000,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/NousResearch/Nous-Hermes-13B/blob/main/config.json
    dict(
        org="NousResearch",
        name="Nous-Hermes-13b",
        block_size=2048,
        vocab_size=32000,
        padded_vocab_size=32001,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b
    dict(
        org="NousResearch",
        name="Nous-Hermes-Llama2-13b",
        vocab_size=32000,
        padded_vocab_size=32032,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
]
configs.extend(nous_research)

###############
# Meta LLaMA 2
###############
llama_2 = [
    # https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
    dict(
        org="meta-llama",
        name="Llama-2-7b{}-hf",
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
    dict(
        org="meta-llama",
        name="Llama-2-13b{}-hf",
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json
    dict(
        org="meta-llama",
        name="Llama-2-70b{}-hf",
        vocab_size=32000,
        padding_multiple=64,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
    ),   
     dict(
        org="meta-llama",
        name="Llama-2-49M{}-hf",
        vocab_size=35000,
        padding_multiple=64,
        block_size=2048,
        n_layer=5,
        n_head=5,
        n_embd=500,
        n_query_groups=5,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,        
        norm_eps=1.0e-8,
    ),
    dict(
        org="meta-llama",
        name="Llama-2-130M{}-hf",
        vocab_size=35000,
        padding_multiple=64,
        block_size=2048,
        n_layer=10,
        n_head=12,
        n_embd=720,
        n_query_groups=6,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=3000,
        norm_eps=1.0e-6,
        nef=True
    ),
    dict(
        org="meta-llama",
        name="Llama-2-130M_v2{}-hf",
        vocab_size=35000,
        padding_multiple=64,
        block_size=2048,
        n_layer=14,
        n_head=16,
        n_embd=960,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=128,
        norm_eps=1.0e-6,
    ),
    dict(
        org="meta-llama",
        name="Llama-2-250M{}-hf",
        vocab_size=35000,
        padding_multiple=64,
        block_size=2048,
        n_layer=16,
        n_head=16,
        n_embd=1024,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=2000,
        norm_eps=1.0e-6,
    ),
    dict(
        org="meta-llama",
        name="Llama-2-350M{}-hf",
        vocab_size=35000,
        padding_multiple=64,
        block_size=2048,
        n_layer=10,
        n_head=10,
        n_embd=800,
        n_query_groups=5,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,        
        norm_eps=1.0e-5,
    ),
    dict(
        org="meta-llama",
        name="Llama-2-350M_v2{}-hf",
        vocab_size=35000,
        padding_multiple=64,
        block_size=2048,
        n_layer=14,
        n_head=14,
        n_embd=1260,
        n_query_groups=7,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=4000,
        norm_eps=1.0e-8,
    ),
    dict(
        org="meta-llama",
        name="Llama-2-400M{}-hf",
        vocab_size=35000,
        padding_multiple=64,
        block_size=2048,
        n_layer=12,
        n_head=12,
        n_embd=840,
        n_query_groups=6,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        norm_eps=1.0e-8,
        nef=False,
        _description="huggingfaceにupload済 v1",
    ),
     dict(
        org="meta-llama",
        name="Llama-2-400M{}_v3-hf",
        vocab_size=35000,
        padding_multiple=64,
        block_size=2048,
        n_layer=14,
        n_head=14,
        n_embd=980,
        n_query_groups=7,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=8000,
        norm_eps=1.0e-6,
        _description="loss 3.5まで減少一旦保存",
    ),
    dict(
        org="meta-llama",
        name="Llama-2-400M{}_v4-hf",
        vocab_size=35000,
        padding_multiple=64,
        block_size=2048,
        n_layer=22,
        n_head=14,
        n_embd=980,
        n_query_groups=7,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=4400,
        norm_eps=1.0e-6,
        _description="416M head数少ないパターン",
    ),
    dict(
        org="meta-llama",
        name="Llama-2-400M{}_v5-hf",
        vocab_size=35000,
        padding_multiple=64,
        block_size=2048,
        n_layer=4,
        n_head=24,
        n_embd=2880,
        n_query_groups=12,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=5000,
        norm_eps=1.0e-6,
        _description="474M layer数少ないパターン",
    ),
    dict(
        org="meta-llama",
        name="Llama-2-550M{}-hf",
        vocab_size=35000,
        padding_multiple=64,
        block_size=2048,
        n_layer=14,
        n_head=16,
        n_embd=960,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,        
        norm_eps=1.0e-8,
    ),
    dict(
        org="meta-llama",
        name="Llama-2-600M{}-hf",
        vocab_size=35000,
        padding_multiple=64,
        block_size=2048,
        n_layer=18,
        n_head=16,
        n_embd=960,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,        
        norm_eps=1.0e-8,
    ),
    dict(
        org="meta-llama",
        name="Llama-2-950M{}-hf",
        vocab_size=35000,
        block_size=2048,
        padding_multiple=64,
        n_layer=24,
        n_head=16,
        n_embd=1024,
        n_query_groups=4,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
]
for c in llama_2:
    for kind in ("", "-chat"):
        copy = c.copy()
        copy["name"] = c["name"].format(kind)
        configs.append(copy)


# llama = [
#      dict(
#         org="meta-llama",
#         name="Llama-130M",
#         vocab_size=35000,
#         padding_multiple=64,
#         block_size=2048,
#         n_layer=12,
#         n_head=10,
#         n_embd=780,
#         n_query_groups=10,
#         rotary_percentage=1.0,
#         parallel_residual=False,
#         bias=False,
#         _norm_class="RMSNorm",
#         _mlp_class="GptNeoxMLP",
#         intermediate_size=1024,
#         norm_eps=1.0e-6,
#     ),
# ]
# for c in llama:
#     configs.append(c)


##########################
# Stability AI FreeWilly2
##########################
freewilly_2 = [
    # https://huggingface.co/stabilityai/FreeWilly2/blob/main/config.json
    dict(
        org="stabilityai",
        name="FreeWilly2",
        vocab_size=32000,
        padding_multiple=64,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
    )
]
configs.extend(freewilly_2)


##################
# Meta Code Llama
##################
code_llama = [
    # https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json
    dict(
        org="codellama",
        name="CodeLlama-7b-hf",
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-hf/blob/main/config.json
    dict(
        org="codellama",
        name="CodeLlama-13b-hf",
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-hf/blob/main/config.json
    dict(
        org="codellama",
        name="CodeLlama-34b-hf",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-7b-Python-hf/blob/main/config.json
    dict(
        org="codellama",
        name="CodeLlama-7b-Python-hf",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-Python-hf/blob/main/config.json
    dict(
        org="codellama",
        name="CodeLlama-13b-Python-hf",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-Python-hf/blob/main/config.json
    dict(
        org="codellama",
        name="CodeLlama-34b-Python-hf",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/tree/main/config.json
    dict(
        org="codellama",
        name="CodeLlama-7b-Instruct-hf",
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf/blob/main/config.json
    dict(
        org="codellama",
        name="CodeLlama-13b-Instruct-hf",
        block_size=2048,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/blob/main/config.json
    dict(
        org="codellama",
        name="CodeLlama-34b-Instruct-hf",
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
]
configs.extend(code_llama)


########################
# garage-bAInd Platypus
########################
platypus = [
    # https://huggingface.co/garage-bAInd/Platypus-30B/blob/main/config.json
    dict(
        org="garage-bAInd",
        name="Platypus-30B",
        block_size=2048,
        padded_vocab_size=32000,
        n_layer=60,
        n_head=52,
        n_embd=6656,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-06,
        _mlp_class="LLaMAMLP",
        intermediate_size=17920,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-7B/blob/main/config.json
    dict(
        org="garage-bAInd",
        name="Platypus2-7B",
        padded_vocab_size=32000,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-13B/blob/main/config.json
    dict(
        org="garage-bAInd",
        name="Platypus2-13B",
        padded_vocab_size=32000,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-70B/blob/main/config.json
    dict(
        org="garage-bAInd",
        name="Platypus2-70B",
        padded_vocab_size=32000,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
    ),
    # https://huggingface.co/garage-bAInd/Camel-Platypus2-13B/blob/main/config.json
    dict(
        org="garage-bAInd",
        name="Camel-Platypus2-13B",
        padded_vocab_size=32000,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/garage-bAInd/Camel-Platypus2-70B/blob/main/config.json
    dict(
        org="garage-bAInd",
        name="Camel-Platypus2-70B",
        padded_vocab_size=32000,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
    ),
    # https://huggingface.co/garage-bAInd/Stable-Platypus2-13B/blob/main/config.json
    dict(
        org="garage-bAInd",
        name="Stable-Platypus2-13B",
        padded_vocab_size=32000,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-70B-instruct/blob/main/config.json
    dict(
        org="garage-bAInd",
        name="Platypus2-70B-instruct",
        padded_vocab_size=32000,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
    ),
]
configs.extend(platypus)


##########################
# Stability AI StableCode
##########################
stablecode = [
    # https://huggingface.co/stabilityai/stablecode-completion-alpha-3b/blob/main/config.json
    dict(
        org="stabilityai",
        name="stablecode-completion-alpha-3b",
        block_size=16384,
        vocab_size=49152,
        n_layer=32,
        n_embd=2560,
    ),
    # https://huggingface.co/stabilityai/stablecode-completion-alpha-3b-4k/blob/main/config.json
    dict(org="stabilityai", name="stablecode-completion-alpha-3b-4k", vocab_size=49152, n_layer=32, n_embd=2560),
    # https://huggingface.co/stabilityai/stablecode-instruct-alpha-3b/blob/main/config.json
    dict(org="stabilityai", name="stablecode-instruct-alpha-3b", vocab_size=49152, n_layer=32, n_embd=2560),
]
configs.extend(stablecode)


##################################
# togethercomputer LLaMA-2-7B-32K
##################################
together_llama2_32k = [
    # https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/blob/main/config.json
    dict(
        org="togethercomputer",
        name="LLaMA-2-7B-32K",
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_condense_ratio=8,
    )
]
configs.extend(together_llama2_32k)


################
# Microsoft Phi
################
phi = [
    # https://huggingface.co/microsoft/phi-1_5/blob/main/config.json
    dict(
        org="microsoft",
        name="phi-1_5",
        vocab_size=50257,
        padded_vocab_size=51200,
        block_size=2048,
        n_embd=2048,
        n_layer=24,
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
    ),
    dict(
        org="microsoft",
        name="phi-1_5-small",
        vocab_size=35000,
        padded_vocab_size=35000,        
        block_size=2048,        
        n_layer=10,
        n_head=12,
        n_embd=720,
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
    ),
    dict(
        org="microsoft",
        name="phi-1_5-130M",
        vocab_size=35000,
        padded_vocab_size=35000,
        block_size=2048,
        n_layer=12,
        n_head=10,
        n_embd=780,
        n_query_groups=10,
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
    ),
    dict(
        org="microsoft",
        name="phi-1_5-130M_v2",
        vocab_size=35000,
        padded_vocab_size=35000,
        block_size=2048,
        n_layer=12,
        n_head=10,
        n_embd=800,
        n_query_groups=10,
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        shared_attention_norm=True,
        parallel_residual=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
        intermediate_size=1024,
        norm_eps=1e-5
        # norm_eps=1e-6
    ),
    dict(
        org="microsoft",
        name="phi-1_5-350M",
        vocab_size=35000,
        padded_vocab_size=35000,
        block_size=2048,
        n_layer=20,
        n_head=24,
        n_embd=1056,
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
    ),
    dict(
        org="microsoft",
        name="phi-1_5-400M",
        vocab_size=35000,
        padded_vocab_size=35000,
        block_size=2048,
        n_layer=10,
        n_head=16,
        n_embd=1600,
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
        _description="419M iter: 450559,  val loss: 3.6",
    ),
    dict(
        org="microsoft",
        name="phi-1_5-400M_deep_layer",
        vocab_size=35000,
        padded_vocab_size=35000,
        block_size=2048,
        n_layer=14,
        n_head=1,
        n_embd=1400,
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
        _description="427.53M",
    ),
    dict(
        org="microsoft",
        name="phi-1_5-400M_multi_head",
        vocab_size=35000,
        padded_vocab_size=35000,
        block_size=2048,
        n_layer=1,
        n_head=36,
        n_embd=3600,
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
        _description="407.60M",
    ),
    dict(
        org="microsoft",
        name="phi-1_5-400M_auto_encoder",
        vocab_size=35000,
        padded_vocab_size=35000,
        block_size=2048,
        n_layer=-1,
        n_head=1,
        _n_embd=[4000, 4000, 4000, 4000, 2000, 2000, 2000, 1000, 500, 1000, 2000, 2000, 2000, 4000, 4000, 4000, 4000],
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
        _description="420.05M",
    )
]
configs.extend(phi)

name_to_config = {config["name"]: config for config in configs}