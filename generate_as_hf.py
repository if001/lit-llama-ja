import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import HFTokenizer
from lit_llama.utils import lazy_load, llama_model_lookup, quantization
from mixtral_hf.mixtral import MixtralConfig_HF, MixtralForCausalLM_HF
from mixtral_hf.traning_config import TrainingConfig

from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

def main(
    prompt: str = "Hello, my name is",
    *,
    num_samples: int = 1,
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
    use_mixtral_moe: bool = False,
    tokenizer_as_pretrained: bool = False,
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
    if not tokenizer_as_pretrained:
        assert tokenizer_path.is_file(), tokenizer_path

    print(f"Loading model ...{model_name}", file=sys.stderr)
    t0 = time.time()
    with lazy_load(checkpoint_path) as checkpoint:
        # name = llama_model_lookup(checkpoint)        
        config = MixtralConfig_HF.from_name(model_name)
        # model = MixtralForCausalLM_HF(config)
        model = MixtralForCausalLM(config)
        model.load_state_dict(checkpoint)
        model.config.output_router_logits=False
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()    
    tokenizer = HFTokenizer(tokenizer_path, as_pretrained=tokenizer_as_pretrained)
    encoded = tokenizer.encode(prompt, bos=True, eos=False)
    encoded = encoded.to(dtype=torch.int64)
    encoded = encoded.unsqueeze(0)
    print('encoded', encoded)
    result = model.generate(
        encoded,
        num_beams=1,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    print(tokenizer.decode(result[0]))

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
