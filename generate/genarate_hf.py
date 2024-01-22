import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Union

import lightning as L
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers.generation.utils import (
    LogitsProcessorList, 
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
    RepetitionPenaltyLogitsProcessor
)

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from lit_llama import GPT
from mixtral_hf import MixtralForCausalLM_HF

@torch.no_grad()
def generate(
    model: Union[GPT, MixtralForCausalLM_HF],
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    max_seq_length: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,

    eos_id: Optional[int] = None,
    use_mixtral_moe: bool = False
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
    """

    logits_processor = LogitsProcessorList([
        RepetitionPenaltyLogitsProcessor(repetition_penalty),
    ])

    logits_wraper = LogitsProcessorList([
            TemperatureLogitsWarper(temperature),
            TopPLogitsWarper(top_p),
            TopKLogitsWarper(top_k),            
    ])
    # create an empty tensor of the expected final shape and fill in the current tokens    
    T = idx.size(0)
    T_new = T + max_new_tokens
    if hasattr(model.config, 'block_size'):
        block_size = model.config.block_size 
    else:
        block_size = 2048    
    if max_seq_length is None:
        max_seq_length = min(T_new, block_size)

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    # empty = torch.empty(T_new, dtype=dtype, device=device)
    empty = torch.zeros(T_new, dtype=dtype, device=device)
    empty[:T] = idx    
    idx = empty

    input_pos = torch.arange(0, T, device=device)

    if idx.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    # generate max_new_tokens tokens
    for _ in range(max_new_tokens):
        x = idx.index_select(0, input_pos).view(1, -1).to(dtype=torch.int64)        

        # forward
        if use_mixtral_moe:
            logits, _ = model(x, input_pos)
        else:
            logits = model(x, input_pos)
        # logits = logits[0, -1]
        logits = logits[:, -1, :]
        next_token_scores = logits_processor(x, logits)
        next_token_scores = logits_wraper(x, next_token_scores)
        next_token_scores = next_token_scores.squeeze(0)
        
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)        
        idx_next = idx_next.to(dtype=dtype)

        # advance
        # input_pos = input_pos[-1:] + 1
        last_pos = input_pos[-1].unsqueeze(0) + 1
        input_pos = torch.cat((input_pos, last_pos))

        if idx.device.type == "xla":
            xm.mark_step()

        # concatenate the new generation
        # idx = idx.index_copy(0, input_pos, idx_next)
        idx = idx.index_copy(0, last_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            # return idx[:input_pos]  # include the EOS token
            return idx[:last_pos]  # include the EOS token

    return idx
