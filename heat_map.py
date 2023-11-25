import time
import sys
from pathlib import Path
from typing import Optional

import seaborn as sns
import matplotlib.pyplot as plt
import lightning as L
import torch

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import LLaMA, Tokenizer, HFTokenizer
from lit_llama import GPT
from lit_llama.utils import lazy_load, llama_model_lookup, quantization

@torch.no_grad()
def gen(
        model: GPT,
        idx: torch.Tensor,
):
    print('idx, ', idx.shape)
    attention = None
    def hook_function(module, input, output):
        nonlocal attention
        print("attention 0, ", output.shape, output)
        attention = output

    device, dtype = idx.device, idx.dtype
    T = idx.size(0)
    input_pos = torch.arange(0, T, device=device)

    x = idx.index_select(0, input_pos).view(1, -1).to(dtype=torch.int64)
    
    last_layer = model.config.n_layer
    print('1: ', model.transformer.h)
    print('2: ', model.transformer.h[last_layer])
    print('3: ',model.transformer.h[last_layer][0])

    hook = model.transformer.h[last_layer][0].atten.register_forward_hook(hook_function)
    model(x, input_pos)
    hook.remove()

    print("attention, ", attention.shape, attention)
    return attention

def main(
        prompt: str = "",
        model_name: str = "",
        checkpoint_path: str = "",
        tokenizer_path: str = "",
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
    gen(model, encoded)


    # outputs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])
    # attention = torch.mean(outputs.attentions[-1], dim=1)[0].detach().numpy()

    # ヒートマップの作成
    # sns.heatmap(attention, cmap="YlGnBu", xticklabels=tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]), yticklabels=tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]))
    # plt.show()

if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)