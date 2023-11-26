import time
import sys
from pathlib import Path
from typing import Optional

import seaborn as sns
import matplotlib.pyplot as plt
import lightning as L
import torch
import numpy as np

## matplotlibのlegendに日本語を使う
import japanize_matplotlib
japanize_matplotlib.japanize()


wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import LLaMA, Tokenizer, HFTokenizer
from lit_llama import GPT
from lit_llama.utils import lazy_load, llama_model_lookup, quantization

@torch.no_grad()
def gen(
        model: GPT,
        idx: torch.Tensor,
        target_layer_idx: Optional[int],
):
    print('idx, ', idx.shape)
    attention = None
    def hook_function(module, input, output):
        nonlocal attention
        _, attention = output
    
    device, dtype = idx.device, idx.dtype
    T = idx.size(0)
    input_pos = torch.arange(0, T, device=device)

    x = idx.index_select(0, input_pos).view(1, -1).to(dtype=torch.int64)
    
    if target_layer_idx is None:
        target_idx = model.config.n_layer - 1
    target_idx = target_layer_idx

    hook = model.transformer.h[target_idx].attn.register_forward_hook(hook_function)
    model(x, input_pos)
    hook.remove()
    q, k, v = attention
    
    attention_weights = []
    for k_part, q_part in zip(k[0],q[0]):
        k_part=k_part[:T,]
        _k = torch.transpose(k_part, 0, 1) ## 1, num_heads, seq_len, hidden_dim => hidden_dim, seq_len
        attention_weight = torch.matmul(q_part, _k) / np.sqrt(q_part.size(-1))
        attention_weight = attention_weight.squeeze()
        attention_weights.append(attention_weight)

    return attention_weights

def main(
        prompt: str = "",
        model_name: str = "",
        checkpoint_path: str = "",
        tokenizer_path: str = "",
        target_layer_idx: Optional[int] = None,
        quantize: Optional[str] = None,
):
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
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)
    prompt_length = encoded.size(0)

    # テキストのトークン化    
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)

    
    # Attentionの取得
    attention_weights = gen(model, encoded, target_layer_idx)
    graph_num = len(attention_weights)
    if graph_num == 1:
        figsize=(6, 6*graph_num)
    else:
        figsize=(6*4, 6*2)
    plt.figure(figsize=figsize)    

    for i, attention in enumerate(attention_weights):        
        _c = 1 if i <= 4 else 2 ## col 列
        if graph_num == 1:
            plt.subplot(1, 1, 1)
        else:
            plt.subplot(2, 4, i+1)
        attention = attention.to('cpu').detach().numpy().copy()
        # attention = torch.mean(outputs.attentions[-1], dim=1)[0].detach().numpy()

        # ヒートマップの作成    
        labels = tokenizer.tokenize(prompt)
        labels = ['bos'] + labels
        # print('labels', labels)
        sns.heatmap(attention, cmap="YlGnBu", 
                    xticklabels=labels, 
                    yticklabels=labels)
        plt.title(f"head_{i+1}")
    # plt.savefig("/content/drive/MyDrive/pre_trained/llama2/save.png")
    plt.show()


    # outputs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])
    # attention = torch.mean(outputs.attentions[-1], dim=1)[0].detach().numpy()

    # ヒートマップの作成
    # sns.heatmap(attention, cmap="YlGnBu", xticklabels=tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]), yticklabels=tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]))
    # plt.show()

if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)