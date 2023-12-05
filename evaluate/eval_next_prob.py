import sys
import time
import warnings
from pathlib import Path
from typing import Optional
import re
from tqdm import tqdm

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


texts = """昨夜見た夢について書きたいと思います。夢の中で私は
今日、私は新しいレシピに挑戦しました。材料は以下の通りです
子供の頃の思い出には特別なものがあります。特に覚えているのは
私が最初に海外旅行に行った時のことを思い出します。目的地は
最近読んだ本の中で、特に心に残った一節があります。それは
科学の進歩は驚異的です。特に注目している分野は
自然は驚くべき美しさを秘めています。私のお気に入りの自然現象は
アートは人の心に深く訴えかけます。私が最も影響を受けた作品は
スポーツは私の人生に大きな影響を与えています。特に好きなスポーツは
映画は素晴らしい娯楽です。私のお気に入りの映画は
音楽は私の毎日を彩ります。最近ハマっている曲は
テクノロジーの進歩には目を見張るものがあります。特に興味深い発明は
歴史には学ぶべき教訓がたくさんあります。特に興味深い時代は
私が初めて料理に挑戦した時のことを覚えています。作ったのは
幼い頃、私は特定のテレビ番組に夢中でした。それは
旅行は人生を豊かにします。私が最も印象に残っている旅行先は
私の家族について書きたいと思います。家族構成は
人生で学んだ大切な教訓について共有したいと思います。それは
私の趣味について話したいと思います。最も情熱を注いでいる趣味は
学生時代の思い出は色褪せないものです。特に忘れられない出来事は
今日の仕事で面白いことがありました。それは
私の故郷について話したいと思います。故郷は
健康について意識していることがあります。それは
小さい頃の夢について書きたいと思います。夢は
エコロジーについて考えることが多いです。特に重要だと思うのは
時代の変化を感じる瞬間があります。それは
私が尊敬する人物について話したいと思います。その人は
子供の頃に学んだ大切なことは何ですか？それは
私の一日のルーティンについて書きたいと思います。朝は
ある特別な出来事が私の人生を変えました。それは"""

def get_texts():
    return texts.split('\n')

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
            TopKLogitsWarper(top_k),
            TopPLogitsWarper(top_p),
            TemperatureLogitsWarper(temperature),
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

    # generate max_new_tokens tokens
    for _ in range(max_new_tokens):
        x = idx.index_select(0, input_pos).view(1, -1).to(dtype=torch.int64)        

        # forward
        logits = model(x, input_pos)
        # logits = logits[0, -1]
        logits = logits[:, -1, :]
        next_token_scores = logits_processor(x, logits)
        next_token_scores = logits_wraper(x, next_token_scores)
        next_token_scores = next_token_scores.squeeze(0)
        
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)

        print('probs', probs)

        idx_next = torch.multinomial(probs, num_samples=1)
        idx_next = idx_next.to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1        

        if idx.device.type == "xla":
            xm.mark_step()

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx

def load_ppl_model(model_path, sp_model_path):
    import kenlm
    import sentencepiece    
    model = kenlm.LanguageModel(model_path)
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(sp_model_path)
    return sp, model

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
    y = generate(model, encoded, max_new_tokens, 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p, 
                repetition_penalty=repetition_penalty,
                eos_id=eos_id)
    result_text = tokenizer.decode(y)        
    print(result_text)

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
