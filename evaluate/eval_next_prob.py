"""
入力した文章に続く、単語をtop_k分の確率と単語を表示する

例:
input 
text:  0.4670667350292206 山
text:  0.16267991065979004 水
text:  0.14469051361083984 、
text:  0.12376037240028381 ウ
text:  0.10180250555276871 から
----------------------------------------------------------------------------------------------------
input 私は
text:  0.3446035385131836 「
text:  0.2156473845243454 お
text:  0.16405588388442993 私は
text:  0.14591431617736816 東京
text:  0.1297788769006729 この
----------------------------------------------------------------------------------------------------
input 、
text:  0.657270610332489 、
text:  0.16107109189033508 都
text:  0.07973597198724747 ・
text:  0.03947216644883156 を
text:  0.03122505359351635 や
----------------------------------------------------------------------------------------------------
input 東京
text:  0.5059266686439514 大阪
text:  0.31660082936286926 東京
text:  0.07175566256046295 京都
text:  0.059024590998888016 神
text:  0.046692296862602234 愛
----------------------------------------------------------------------------------------------------
input 、
text:  0.9138786196708679 知
text:  0.04015302658081055 媛
text:  0.0214923657476902 、
text:  0.015724150463938713 川
text:  0.008751808665692806 称
----------------------------------------------------------------------------------------------------
私は、東京、愛、
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Optional
import numpy as np

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
import datasets


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import LLaMA, Tokenizer, HFTokenizer
from lit_llama import GPT
from lit_llama.utils import lazy_load, llama_model_lookup, quantization

import matplotlib.pyplot as plt
import networkx as nx
import japanize_matplotlib
japanize_matplotlib.japanize()


def show_graph(probs, start_prompt, save_fig_name):
    G = nx.DiGraph()
    start=start_prompt
    G.add_node(start)

    current_node = start
    for idx, level in enumerate(probs):
        next_node = repr(max(level, key=lambda x: x['p'])['text'])
        for item in level:
            node_name = f"{repr(item['text'])}_{idx+1}"            
            G.add_node(node_name)
            G.add_edge(current_node, node_name, label=f"{item['p']:.2f}")
        current_node = f"{next_node}_{idx+1}"
    
    # レイアウトを階層的に設定
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    edge_labels = nx.get_edge_attributes(G, 'label')

    # グラフの描画
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=2000, font_size=10, font_weight='bold', font_family='IPAexGothic')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title("")
    # plt.show()
    print('save fig...', save_fig_name)
    plt.savefig(save_fig_name)

@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    max_seq_length: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,

    eos_id: Optional[int] = None,
    use_mixtral_moe: bool = False,
    simple: bool = False,
    greedy: bool = False
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
    if max_seq_length is None:
        max_seq_length = min(T_new, model.config.block_size)    

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
    
    next_probs = []
    current_idxs = []
    # generate max_new_tokens tokens
    for _ in range(max_new_tokens):
        x = idx.index_select(0, input_pos).view(1, -1).to(dtype=torch.int64)        

        if use_mixtral_moe:
            logits, _ = model(x, input_pos)
        else:
            logits = model(x, input_pos)
        # logits = logits[0, -1]
        logits = logits[:, -1, :] ## [1, seq_size, vocab_size] =>  [1, vocab_size]
        if simple:
            next_token_scores = logits
        else:
            next_token_scores = logits_processor(x, logits)
            next_token_scores = logits_wraper(x, next_token_scores)
            next_token_scores = next_token_scores.squeeze(0) ## [1, 35008] => [35008]
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        
        _probs=probs.detach()
        top_values, top_indices = torch.topk(_probs, top_k)        
        next_prob = [{"index": int(index), "p": float(prob)} for index, prob in zip(top_indices, top_values)]
        next_probs.append(next_prob)
        if greedy:
            idx_next = torch.argmax(probs, dim=-1) ## greedy search
        else:
            idx_next = torch.multinomial(probs, num_samples=1) ## sample
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
        current_idxs.append(idx)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            # return idx[:input_pos]  # include the EOS token
            return idx[:last_pos]  # include the EOS token

    return idx, next_probs,current_idxs

def make_graph(probs):
    pass

def main(
    prompt: str = "",
    max_new_tokens: int = 50,
    top_k: int = 200,
    top_p: float = 0.9,
    temperature: float = 0.8,
    repetition_penalty: float = 1.0,
    checkpoint_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = None,
    model_name: str = "7B",
    eos_id: Optional[int] = None,
    kenlm_model_path: str = "",
    sp_model_path: str = "",
    use_mixtral_moe: bool = False,
    simple: bool = False,
    greedy: bool = False
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    assert checkpoint_path.is_file(), checkpoint_path
    assert tokenizer_path.is_file(), tokenizer_path

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print(f"Loading model ...{model_name}", file=sys.stderr)
    t0 = time.time()
    with lazy_load(checkpoint_path) as checkpoint:
        # name = llama_model_lookup(checkpoint)
        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = GPT.from_name(model_name)
        model.load_state_dict(checkpoint)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)
     
    with fabric.init_tensor():
        # enable the kv cache
        model.set_kv_cache(batch_size=1, index=0)
    
    tokenizer = HFTokenizer(tokenizer_path)

    L.seed_everything(1234)
    
    
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)
    result_ids, next_probs, current_idxs = generate(model, encoded, max_new_tokens, 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p, 
                repetition_penalty=repetition_penalty,
                eos_id=eos_id,
                use_mixtral_moe=use_mixtral_moe,
                simple=simple,
                greedy=greedy
                )
    
    new_probs = []
    for ids, probs in zip(current_idxs, next_probs):
        input = tokenizer.decode(ids)
        print('input', input)
        nodes = []
        for p in probs:
            text = tokenizer.decode(torch.tensor([p['index']]))
            _p = float(p['p'])
            print(f'p:{_p:.2f}, {text}')
            nodes.append({ 'text': text, 'p': p['p']})
        new_probs.append(nodes)
        print('-'*100)
    
    save_fig_name = f"./tree_{model_name}.png"
    show_graph(new_probs, prompt, save_fig_name)

    result_text = tokenizer.decode(result_ids)
    print('final result: ', result_text)

    # make_graph(new_probs)

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore", 
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    CLI(main)

