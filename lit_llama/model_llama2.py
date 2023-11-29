"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.functional as F
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self

from lit_llama.config_llama2 import Llama2Config as Config

FlashAttention2Available = bool(RequirementCache("flash-attn>=2.0.0.post1"))


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        print(f'FlashAttention2Available: {FlashAttention2Available}')
        self.config = config

        self._cos_list = []
        self._sin_list = []

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        if config.nef:
            embed = EmbeddingNEFTune(config)
        else:
            ## orignal
            embed = nn.Embedding(config.padded_vocab_size, config.n_embd)
        self.transformer = nn.ModuleDict(
            dict(
                wte=embed,
                h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
        self._max_seq_length = value

        if len(self._cos_list) == 0:
            for i in range(self.config.n_layer):
                cos, sin = self.rope_cache(i)
                self._cos_list.append(cos)
                self._sin_list.append(sin)
                self.register_buffer(f"cos_{i}", cos, persistent=False)
                self.register_buffer(f"sin_{i}", sin, persistent=False)
        else:
            
            for i in range(self.config.n_layer):
                if value != self._cos_list[i].size(0):                    
                    # override
                    cos, sin = self.rope_cache(device=self.cos.device)
                    self._cos_list[i] = cos
                    self._sin_list[i] = cos

        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.max_seq_length = self.config.block_size

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")


        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for i, block in enumerate(self.transformer.h):
            if input_pos is not None:  # use the kv cache
                cos = self._cos_list[i].index_select(0, input_pos)
                sin = self._sin_list[i].index_select(0, input_pos)
                if self.mask_cache is None:
                    raise TypeError("You need to call `gpt.set_kv_cache()`")
                mask = self.mask_cache.index_select(2, input_pos)
            else:            
                cos = self._cos_list[i][:T]
                sin = self._sin_list[i][:T]
                mask = None
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, i: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elems[i],
            dtype=torch.get_default_dtype(),
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        index: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self._cos_list[index].size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
            ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
            self.mask_cache = torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None

class EmbeddingNEFTune(nn.Module):
    """
    Embedding with NEFTune
    https://github.com/neelsjain/NEFTune
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.noise_alpha = 5
        self.org_embed = nn.Embedding(config.padded_vocab_size, config.n_embd)

    def forward(self, input: torch.Tensor) -> torch.Tensor:        
        embed_init = self.org_embed(input)        
        dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
        mag_norm = self.noise_alpha/torch.sqrt(dims)
        return embed_init + torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)

class Block(nn.Module):
    def __init__(self, config: Config, idx=-1) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)        
        self.attn = CausalSelfAttention(config, idx)
        self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_1 = self.norm_1(x)
        h, _ = self.attn(n_1, cos, sin, mask, input_pos)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, idx = 0) -> None:        
        super().__init__()
        self._idx = idx
        self._n_query_groups = config.n_query_groups_list[idx]

        # shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        shape = (config.n_heads[idx] + 2 * self._n_query_groups) * config.head_sizes[idx]        

        self._head_size = config.head_sizes[idx]
        self._n_head = config.n_heads[idx]
        self._rope_n_elem = config.rope_n_elems[idx]

        # key, query, value projections for all heads, but in a batch
        start_dim = config.n_embd
        if config.compress:
            start_dim = config.n_embd/2

        if not config.separate_qkv:
            self.attn = nn.Linear(start_dim, shape, bias=config.bias)
            # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        # self.active = nn.ReLU()
        self.active = lambda x: torch.nn.functional.gelu(x, approximate="tanh")

        self.config = config
        
        if config.separate_qkv:
            q_per_kv = self._n_head // self._n_query_groups
            _q_shape = self._n_query_groups * q_per_kv * self._head_size
            self.q_l = nn.Linear(config.n_embd, _q_shape)

            _kv_shape = self._n_query_groups * self._head_size
            self.k_l = nn.Linear(config.n_embd, _kv_shape)
            self.v_l = nn.Linear(config.n_embd, _kv_shape)

        # if config.use_scale_tensor:
        #    self.scaling = nn.Linear(config.n_embd, config.n_embd)
        self.scaling = nn.Linear(config.n_embd, config.n_embd)
        self.scale_active = lambda x: nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        if not self.config.separate_qkv: ## original: qkvにそれぞれlinearを割り当てない            
            qkv = self.attn(x) ## batch size, sequence length, embedding dimensionality (n_embd)
            if self.config.non_liner or self.config.compress:
                qkv = self.active(qkv)

            # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
            q_per_kv = self._n_head // self._n_query_groups

            total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
            qkv = qkv.view(B, T, self._n_query_groups, total_qkv, self._head_size)
            qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

            # split batched computation into three
            q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            # q (B, _n_query_groups, q_per_kv, T, hs)
            # k (B, _n_query_groups, 1, T, hs)
            # v (B, _n_query_groups, 1, T, hs)

        q_per_kv = self._n_head // self._n_query_groups
        if self.config.separate_qkv: ## original: qkvにそれぞれlinearを割り当てる
            _q = self.q_l(x)
            _q = self.active(_q)
            q = _q.view(B, self._n_query_groups, q_per_kv, T, self._head_size)

            _k = self.k_l(x)
            _k = self.active(_k)
            k = _k.view(B, self._n_query_groups, 1, T, self._head_size)

            _v = self.k_l(x)
            _v = self.active(_v)
            v = _v.view(B, self._n_query_groups, 1, T, self._head_size)

        # repeat k and v if necessary
        if self._n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
            # for MHA this is a no-op
            k = k.expand(B, self._n_query_groups, q_per_kv, T, self._head_size)
            v = v.expand(B, self._n_query_groups, q_per_kv, T, self._head_size)

        q = q.reshape(B, -1, T, self._head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self._head_size)  # (B, nh_k, T, hs)
        
        v = v.reshape(B, -1, T, self._head_size)  # (B, nh_v, T, hs)
        q_roped = apply_rope(q[..., : self._rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self._rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self._rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self._rope_n_elem :]), dim=-1)
        print('k0', k.shape)
        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        # q (B, nh_q, T, hs)
        # k (B, nh_k, block_size, hs)
        # v (B, nh_v, block_size or T, hs)
        print('q', q.shape)
        print('k', k.shape)
        print('k2', k.transpose(-2, -1).shape)
        
        scaling = self.scale_active(self.scaling(x))
        print('q.size', q.size(-2))
        y = self._scaled_dot_product_attention_v2(q, k, v, scaling, mask)
        print('y', y.shape)

        # if self.config.use_scale_tensor:
        #     scaling = self.scaling(x)
        #     scaling = self.active(scaling)          
        #     y = self._scaled_dot_product_attention_v2(q, k, v, scaling, mask)
        # else:
        #     y = self.scaled_dot_product_attention(q, k, v, mask)
        #     y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        if self.config.non_liner or self.config.compress:            
            y = self.active(y)
        return y, (q, k, v)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self._head_size)
        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            # flash-attn requires (B, T, nh, hs)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)
        # y = torch.nn.functional.scaled_dot_product_attention(
        #     q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        # )
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=mask is None
        )
        return y.transpose(1, 2)

    ## torch実装
    def _scaled_dot_product_attention_v2(self, query, key, value, scale_tensor, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:        
        print('query', query)
        print('key', key)
        # Efficient implementation equivalent to the following:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor        
        print('attn_weight', attn_weight)
        print('attn_bias', attn_bias)
        attn_bias = attn_bias.to(device=attn_weight.device)
        print('attn_bias2', attn_bias)
        attn_weight += attn_bias
        attn_weight = attn_weight * scale_tensor ## アダマール積を取ることでscaleする        
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self._n_query_groups == 1 else self._n_head
        v_shape = (batch_size, heads, max_seq_length, self._head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
            k_shape = v_shape
        else:            
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self._head_size - self._rope_n_elem,
            )
            k_shape = v_shape
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v