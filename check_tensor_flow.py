"""
modelに入力を行い、途中の次元数などを確認する用
"""

import sys
from pathlib import Path
from typing import Optional

import lightning as L
import torch

## matplotlibのlegendに日本語を使う
import japanize_matplotlib
japanize_matplotlib.japanize()


wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import HFTokenizer
from lit_llama import GPT
from lit_llama.utils import lazy_load, quantization


def main(
        prompt: str = "",
        model_name: str = "",
        tokenizer_path: str = "",
        quantize: Optional[str] = None,
):
    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print(f"Loading model ...{model_name}", file=sys.stderr)
    with fabric.init_module(empty_init=True), quantization(mode=quantize):
        model = GPT.from_name(model_name)
    model.eval()
    model = fabric.setup(model)
     
    with fabric.init_tensor():
        # enable the kv cache
        model.set_kv_cache(batch_size=1, index=0)

    tokenizer = HFTokenizer(tokenizer_path)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)

    
    device, dtype = encoded.device, encoded.dtype
    T = encoded.size(0)
    input_pos = torch.arange(0, T, device=device)

    x = encoded.index_select(0, input_pos).view(1, -1).to(dtype=torch.int64)
    model(x, input_pos)

if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)